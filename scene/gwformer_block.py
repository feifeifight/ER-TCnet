from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pdc_conv import PDCFusion
from .wavelet_attention import WaveletSpatialAttention


def _resolve_num_heads(dim: int, preferred_heads: int) -> int:
    heads = max(1, min(preferred_heads, dim))
    while dim % heads != 0 and heads > 1:
        heads -= 1
    return heads


def _window_partition(x: torch.Tensor, window_size: int):
    # x: [B, H, W, C]
    b, h, w, c = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = x.permute(0, 3, 1, 2)
        pad_mode = "reflect"
        if pad_h >= h or pad_w >= w:
            pad_mode = "constant"
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
        x = x.permute(0, 2, 3, 1)

    hp = h + pad_h
    wp = w + pad_w

    x = x.view(
        b,
        hp // window_size,
        window_size,
        wp // window_size,
        window_size,
        c,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = x.view(-1, window_size * window_size, c)
    return windows, hp, wp, pad_h, pad_w


def _window_reverse(
    windows: torch.Tensor,
    window_size: int,
    b: int,
    hp: int,
    wp: int,
    h: int,
    w: int,
) -> torch.Tensor:
    c = windows.shape[-1]
    x = windows.view(
        b,
        hp // window_size,
        wp // window_size,
        window_size,
        window_size,
        c,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(b, hp, wp, c)
    x = x[:, :h, :w, :]
    return x


class WindowMSA(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        heads = _resolve_num_heads(dim, num_heads)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x, need_weights=False)
        return y


class PatchMerging(nn.Module):
    def __init__(self, dim: int, out_dim: int = None):
        super().__init__()
        out_dim = out_dim or (2 * dim)
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim)

    def forward(self, x: torch.Tensor, h: int, w: int):
        b, n, c = x.shape
        if n != h * w:
            raise ValueError(f"Token length mismatch, got n={n}, h*w={h*w}")
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"PatchMerging expects even H/W, got {(h, w)}")

        x = x.view(b, h, w, c)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(b, -1, 4 * c)
        x = self.reduction(self.norm(x))
        return x, h // 2, w // 2


class GwformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
        pdc_types: Iterable[str],
        use_wavelet_attention: bool,
        use_window_attention: bool = True,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_window_attention = use_window_attention

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowMSA(dim, num_heads) if use_window_attention else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        if use_wavelet_attention:
            self.wavelet_attn = WaveletSpatialAttention(dim)
        else:
            self.wavelet_attn = nn.Identity()

        self.pdc = PDCFusion(dim, pdc_types)
        self.strip_h = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.strip_v = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.grad_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def _tokens_to_2d(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        if n != h * w:
            raise ValueError(f"Token length mismatch, got n={n}, h*w={h*w}")
        return x.transpose(1, 2).reshape(b, c, h, w)

    def _feat_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)

    def _window_attention(self, x: torch.Tensor, h: int, w: int, shift: bool) -> torch.Tensor:
        b, n, c = x.shape
        window_size = max(1, min(self.window_size, h, w))
        shift_size = self.shift_size if self.shift_size < window_size else 0
        feat = x.view(b, h, w, c)

        if shift and shift_size > 0:
            feat = torch.roll(feat, shifts=(-shift_size, -shift_size), dims=(1, 2))

        windows, hp, wp, _, _ = _window_partition(feat, window_size)
        windows = windows + self.attn(windows)

        feat = _window_reverse(windows, window_size, b, hp, wp, h, w)

        if shift and shift_size > 0:
            feat = torch.roll(feat, shifts=(shift_size, shift_size), dims=(1, 2))

        return feat.view(b, n, c)

    def _strip_pool_gate(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        pooled_h = x.mean(dim=3, keepdim=True)
        pooled_v = x.mean(dim=2, keepdim=True)
        gate_h = self.strip_h(pooled_h).expand(-1, -1, -1, w)
        gate_v = self.strip_v(pooled_v).expand(-1, -1, h, -1)
        gate = torch.sigmoid(gate_h + gate_v)
        return gate

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x_in = x

        # Main branch: LN -> W-MSA/SW-MSA -> residual -> LN -> FFN -> residual -> WSA.
        use_shift = self.shift_size > 0
        x = x + self._window_attention(self.norm1(x), h, w, shift=use_shift)
        x = x + self.mlp(self.norm2(x))

        x2d = self._tokens_to_2d(x, h, w)
        x = x + self._feat_to_tokens(self.wavelet_attn(x2d))

        # Gradient branch: reshape -> PDC -> strip pooling gate -> residual fusion.
        g2d = self._tokens_to_2d(x_in, h, w)
        fc = self.pdc(g2d)
        fg = self.grad_proj(fc) * self._strip_pool_gate(fc)
        x = x + self._feat_to_tokens(fg)

        return x
