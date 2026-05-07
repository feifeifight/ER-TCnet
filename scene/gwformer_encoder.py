from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gwformer_block import GwformerBlock, PatchMerging


class ResBlock(nn.Module):
    """Lightweight residual block for high-resolution feature refinement."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=min(32, dim), num_channels=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=min(32, dim), num_channels=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.silu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.silu(residual + out)


class GWformerEncoder(nn.Module):
    def __init__(
        self,
        img_resolution: int,
        in_channels: int = 3,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        pdc_types: Iterable[str] = ("cpdc", "apdc", "hpdc", "vpdc"),
        use_wavelet_attention: bool = True,
        use_window_attention: bool = True,
    ):
        super().__init__()

        if len(depths) != 4 or len(num_heads) != 4:
            raise ValueError("GWformerEncoder expects 4 stages for depths and num_heads")

        self.img_resolution = img_resolution
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # High-resolution skip connections with residual refinement
        self.init_proj_128 = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, embed_dim), num_channels=embed_dim),
            nn.SiLU(inplace=True),
            ResBlock(embed_dim),
            ResBlock(embed_dim),
        )
        self.init_down_64 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, embed_dim), num_channels=embed_dim),
            nn.SiLU(inplace=True),
            ResBlock(embed_dim),
            ResBlock(embed_dim),
        )

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.patch_norm = nn.LayerNorm(embed_dim)

        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        self.stage_dims = dims

        self.stage_resolutions = [
            img_resolution,
            img_resolution // 2,
            img_resolution // patch_size,
            img_resolution // (patch_size * 2),
            img_resolution // (patch_size * 4),
            img_resolution // (patch_size * 8),
        ]
        self.stage_channels = [
            embed_dim,
            embed_dim,
            embed_dim,
            embed_dim * 2,
            embed_dim * 4,
            embed_dim * 8,
        ]

        self.stages = nn.ModuleList()
        self.patch_merges = nn.ModuleList()

        for stage_idx in range(4):
            blocks = nn.ModuleList()
            for block_idx in range(depths[stage_idx]):
                shift = 0 if block_idx % 2 == 0 else window_size // 2
                blocks.append(
                    GwformerBlock(
                        dim=dims[stage_idx],
                        num_heads=num_heads[stage_idx],
                        window_size=window_size,
                        shift_size=shift,
                        mlp_ratio=mlp_ratio,
                        pdc_types=pdc_types,
                        use_wavelet_attention=use_wavelet_attention,
                        use_window_attention=use_window_attention,
                    )
                )
            self.stages.append(blocks)

            if stage_idx < 3:
                self.patch_merges.append(PatchMerging(dim=dims[stage_idx], out_dim=dims[stage_idx + 1]))

    def forward(self, x: torch.Tensor, point_cloud=None) -> List[torch.Tensor]:
        _ = point_cloud

        x_128 = self.init_proj_128(x)
        x_64 = self.init_down_64(x_128)

        x = self.patch_embed(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.patch_norm(tokens)

        features: List[torch.Tensor] = [x_128, x_64]
        for stage_idx, blocks in enumerate(self.stages):
            for block in blocks:
                tokens = block(tokens, h, w)

            feat = tokens.transpose(1, 2).reshape(b, self.stage_dims[stage_idx], h, w)
            features.append(feat)

            if stage_idx < len(self.patch_merges):
                tokens, h, w = self.patch_merges[stage_idx](tokens, h, w)

        return features
