"""Stage-2 Hindsight 3D Goal Mapper and hierarchical upward goal propagation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from point_hjepa.models.utonia_backbone import FourierPositionalEncoding3D


class GoalPointEncoder(nn.Module):
    """3D Fourier PE (L=6 -> 39D) + 2-layer MLP ``e_psi(p_g)`` (39 -> 256 -> 256)."""

    def __init__(self, num_bands: int = 6, out_dim: int = 256) -> None:
        super().__init__()
        self.pe = FourierPositionalEncoding3D(num_bands=num_bands, include_input=True)
        in_dim = self.pe.out_dim  # 39 for num_bands=6
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, p_g: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pe(p_g))


class _ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class GoalProjectionMapper(nn.Module):
    """3-layer residual MLP ``m_xi([z_t^(1), e_psi(p_g)])`` (512 -> 512 -> 256)."""

    def __init__(self, state_dim: int = 256, hidden_dim: int = 512) -> None:
        super().__init__()
        self.in_proj = nn.Linear(state_dim * 2, hidden_dim)
        self.res1 = _ResidualMLPBlock(hidden_dim)
        self.res2 = _ResidualMLPBlock(hidden_dim)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, z_t1: torch.Tensor, e_g: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(torch.cat([z_t1, e_g], dim=-1))
        h = self.res1(h)
        h = self.res2(h)
        return self.out_proj(h)


class HierarchicalGoalMapper(nn.Module):
    """Maps current Level-1 state + 3D goal point to Level-1..Level-L latent goals."""

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 512,
        fourier_bands: int = 6,
    ) -> None:
        super().__init__()
        self.point_encoder = GoalPointEncoder(num_bands=fourier_bands, out_dim=state_dim)
        self.mapper = GoalProjectionMapper(state_dim=state_dim, hidden_dim=hidden_dim)

    def forward(self, z_t1: torch.Tensor, p_g: torch.Tensor) -> torch.Tensor:
        e_g = self.point_encoder(p_g)
        return self.mapper(z_t1, e_g)

    @torch.no_grad()
    def propagate_goals(
        self,
        z_01: torch.Tensor,
        p_g: torch.Tensor,
        upper_encoders: Sequence[nn.Module],
    ) -> list[torch.Tensor]:
        """Compute ``g_hat^(1)`` and propagate upward through frozen encoders ``E^(l)``."""
        g_curr = self.forward(z_01, p_g)
        goals = [g_curr]
        for enc in upper_encoders:
            g_curr = enc(g_curr)
            goals.append(g_curr)
        return goals
