"""Utonia Tiny perception backbone wrapper with selective stage freezing and 3D Fourier PE."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from typing import Any

import point_transformer_v3 as ptv3
import torch
from point_transformer_v3.model import PointTransformerV3
from torch import nn


class FourierPositionalEncoding3D(nn.Module):
    """Deterministic 3D sinusoidal Fourier positional encoding.

    For ``num_bands=8, include_input=False`` -> 48 dims (``3 * 2 * 8``).
    For ``num_bands=6, include_input=True`` -> 39 dims (``3 + 3 * 2 * 6``).
    """

    def __init__(self, num_bands: int = 8, include_input: bool = False) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        self.include_input = bool(include_input)
        freqs = (2.0 ** torch.arange(self.num_bands, dtype=torch.float32)) * torch.pi
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        base = 3 if self.include_input else 0
        return base + 3 * 2 * self.num_bands

    def forward(self, coord: torch.Tensor) -> torch.Tensor:
        # Compute trig angles in fp32 for positional stability.
        c = coord.float()
        angles = c.unsqueeze(-1) * self.freqs.view(1, 1, -1)  # [M, 3, L]
        sin_feat = torch.sin(angles).reshape(c.shape[0], -1)
        cos_feat = torch.cos(angles).reshape(c.shape[0], -1)
        parts = [c, sin_feat, cos_feat] if self.include_input else [sin_feat, cos_feat]
        return torch.cat(parts, dim=-1)


class UtoniaBackbone(nn.Module):
    """Wraps ``utonia_tiny`` (Stages 1-3 frozen, Stages 4-5 unfrozen) + 3D PE projection."""

    def __init__(
        self,
        preset: str = "utonia_tiny",
        pretrained: bool = True,
        serialization_depth: int = 12,
        freeze_stages: int = 3,
        out_dim: int = 256,
        fourier_bands: int = 8,
    ) -> None:
        super().__init__()
        self.preset = preset
        self.freeze_stages = int(freeze_stages)

        if pretrained:
            self.backbone = ptv3.load(preset, warmup=False, device="cpu", precision="auto")
        else:
            self.backbone = PointTransformerV3.from_preset(preset)

        self.backbone.cfg = dataclasses.replace(
            self.backbone.cfg,
            serialization_depth=int(serialization_depth),
        )
        self._apply_freeze_policy()

        backbone_out_dim = int(self.backbone.out_channels)  # 216 for utonia_tiny
        self.pos_enc = FourierPositionalEncoding3D(num_bands=fourier_bands, include_input=False)
        proj_in_dim = backbone_out_dim + self.pos_enc.out_dim  # 216 + 48 = 264
        self.proj = nn.Sequential(
            nn.Linear(proj_in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def _apply_freeze_policy(self) -> None:
        for param in self.backbone.embedding.parameters():
            param.requires_grad = False
        self.backbone.embedding.eval()

        for idx, stage in enumerate(self.backbone.enc):
            trainable = idx >= self.freeze_stages
            for param in stage.parameters():
                param.requires_grad = trainable
            if not trainable:
                stage.eval()

    def train(self, mode: bool = True) -> "UtoniaBackbone":
        super().train(mode)
        # Keep frozen stem and Stages 1-3 permanently in eval mode (no DropPath stochasticity).
        self.backbone.embedding.eval()
        for idx in range(min(self.freeze_stages, len(self.backbone.enc))):
            self.backbone.enc[idx].eval()
        return self

    def unfrozen_backbone_parameters(self) -> Iterator[nn.Parameter]:
        for idx in range(self.freeze_stages, len(self.backbone.enc)):
            yield from self.backbone.enc[idx].parameters()

    def projection_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.proj.parameters()

    def forward(self, point_data: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run PTv3/Utonia encoder and append 48D Fourier PE -> 256D tokens.

        Returns:
            tokens: ``[M_total, 256]``
            coord: ``[M_total, 3]``
            offset: ``[B * T]`` cumulative token ends per frame.
        """
        out = self.backbone(point_data)
        feat = out["feat"].float()
        coord = out["coord"]
        offset = out["offset"]

        pe = self.pos_enc(coord)
        tokens = self.proj(torch.cat([feat, pe], dim=-1))
        return tokens, coord, offset
