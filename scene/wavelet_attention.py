import torch
import torch.nn as nn
import torch.nn.functional as F


class _SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class WaveletSpatialAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()

        haar = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],   # LL
                [[-1.0, -1.0], [1.0, 1.0]], # LH
                [[-1.0, 1.0], [-1.0, 1.0]], # HL
                [[1.0, -1.0], [-1.0, 1.0]], # HH
            ],
            dtype=torch.float32,
        ) / 4.0

        self.register_buffer("analysis_kernels", haar.unsqueeze(1))
        self.register_buffer("synthesis_kernels", haar.unsqueeze(1))

        self.attn_ll = _SEBlock(channels, reduction)
        self.attn_lh = _SEBlock(channels, reduction)
        self.attn_hl = _SEBlock(channels, reduction)
        self.attn_hh = _SEBlock(channels, reduction)

    def _dwt2d(self, x: torch.Tensor):
        b, c, h, w = x.shape
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"WaveletSpatialAttention requires even H/W, got {(h, w)}")

        weight = self.analysis_kernels.repeat(c, 1, 1, 1)
        y = F.conv2d(x, weight, stride=2, padding=0, groups=c)
        y = y.view(b, c, 4, h // 2, w // 2)
        ll = y[:, :, 0]
        lh = y[:, :, 1]
        hl = y[:, :, 2]
        hh = y[:, :, 3]
        return ll, lh, hl, hh

    def _idwt2d(self, ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
        c = ll.shape[1]
        k = self.synthesis_kernels
        ll_w = k[0:1].repeat(c, 1, 1, 1)
        lh_w = k[1:2].repeat(c, 1, 1, 1)
        hl_w = k[2:3].repeat(c, 1, 1, 1)
        hh_w = k[3:4].repeat(c, 1, 1, 1)

        out = F.conv_transpose2d(ll, ll_w, stride=2, padding=0, groups=c)
        out = out + F.conv_transpose2d(lh, lh_w, stride=2, padding=0, groups=c)
        out = out + F.conv_transpose2d(hl, hl_w, stride=2, padding=0, groups=c)
        out = out + F.conv_transpose2d(hh, hh_w, stride=2, padding=0, groups=c)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = self._dwt2d(x)

        ll = self.attn_ll(ll)
        lh = self.attn_lh(lh)
        hl = self.attn_hl(hl)
        hh = self.attn_hh(hh)

        out = self._idwt2d(ll, lh, hl, hh)
        return x + out
