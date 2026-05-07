import math
from typing import Iterable, List

import torch
import torch.nn as nn


def _make_sparse_pattern(name: str) -> torch.Tensor:
    name = name.lower()
    if name == "cpdc":
        # Center-to-neighbor difference pattern.
        pattern = torch.tensor(
            [[1.0, 1.0, 1.0],
             [1.0, -8.0, 1.0],
             [1.0, 1.0, 1.0]],
            dtype=torch.float32,
        )
    elif name == "apdc":
        # Clock-wise angular difference approximation.
        pattern = torch.tensor(
            [[-1.0, -1.0, 0.0],
             [1.0, 0.0, -1.0],
             [0.0, 1.0, 1.0]],
            dtype=torch.float32,
        )
    elif name == "hpdc":
        pattern = torch.tensor(
            [[1.0, 0.0, -1.0],
             [1.0, 0.0, -1.0],
             [1.0, 0.0, -1.0]],
            dtype=torch.float32,
        )
    elif name == "vpdc":
        pattern = torch.tensor(
            [[1.0, 1.0, 1.0],
             [0.0, 0.0, 0.0],
             [-1.0, -1.0, -1.0]],
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unknown PDC type: {name}")

    pattern = pattern / pattern.abs().sum().clamp(min=1.0)
    return pattern


class _BasePDC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pattern_name: str,
        bias: bool = False,
    ):
        super().__init__()
        self.pattern_name = pattern_name
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        base = _make_sparse_pattern(pattern_name).view(1, 1, 3, 3)
        self.register_buffer("base_pattern", base)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            pattern = self.base_pattern.expand_as(self.conv.weight)
            self.conv.weight.copy_(pattern + 0.01 * self.conv.weight)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CPDC(_BasePDC):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__(in_channels, out_channels, pattern_name="cpdc", bias=bias)


class APDC(_BasePDC):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__(in_channels, out_channels, pattern_name="apdc", bias=bias)


class HPDC(_BasePDC):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__(in_channels, out_channels, pattern_name="hpdc", bias=bias)


class VPDC(_BasePDC):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__(in_channels, out_channels, pattern_name="vpdc", bias=bias)


def _sanitize_pdc_types(pdc_types: Iterable[str]) -> List[str]:
    allowed = {"cpdc", "apdc", "hpdc", "vpdc"}
    out: List[str] = []
    for p in pdc_types:
        lp = p.lower()
        if lp not in allowed:
            raise ValueError(f"Unsupported pdc type: {p}")
        if lp not in out:
            out.append(lp)
    if not out:
        raise ValueError("At least one pdc type must be enabled")
    return out


class PDCFusion(nn.Module):
    def __init__(self, channels: int, pdc_types: Iterable[str]):
        super().__init__()
        pdc_types = _sanitize_pdc_types(pdc_types)
        self.pdc_types = pdc_types

        branches = {}
        for pdc_type in pdc_types:
            if pdc_type == "cpdc":
                branches[pdc_type] = CPDC(channels, channels)
            elif pdc_type == "apdc":
                branches[pdc_type] = APDC(channels, channels)
            elif pdc_type == "hpdc":
                branches[pdc_type] = HPDC(channels, channels)
            elif pdc_type == "vpdc":
                branches[pdc_type] = VPDC(channels, channels)
        self.branches = nn.ModuleDict(branches)

        cat_channels = channels * len(pdc_types)
        self.mix = nn.Conv2d(cat_channels, channels, kernel_size=1, bias=True)
        self.gate = nn.Conv2d(cat_channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.branches[name](x) for name in self.pdc_types]
        cat = torch.cat(feats, dim=1)
        mixed = self.mix(cat)
        gated = torch.sigmoid(self.gate(cat))
        return mixed * gated
