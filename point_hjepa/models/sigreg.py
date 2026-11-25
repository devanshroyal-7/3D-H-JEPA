"""DDP-aware Sketched Isotropic Gaussian Regularization (SIGReg) via Epps-Pulley test."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


def _differentiable_all_gather(x: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return x
    # Gathers tensors along rank list with autograd support back to local rank's slice.
    from torch.distributed.nn.functional import all_gather

    gathered = all_gather(x)
    return torch.cat(gathered, dim=0)


class SIGReg(nn.Module):
    """Epps-Pulley empirical characteristic function test over random 1D unit projections."""

    def __init__(
        self,
        num_slices: int = 1024,
        t_min: float = 0.2,
        t_max: float = 4.0,
        num_knots: int = 32,
    ) -> None:
        super().__init__()
        self.num_slices = int(num_slices)

        t_knots = torch.linspace(t_min, t_max, num_knots, dtype=torch.float32)
        dt = t_knots[1] - t_knots[0]
        weights = torch.full_like(t_knots, dt)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        target_cf = torch.exp(-0.5 * (t_knots**2))

        self.register_buffer("t_knots", t_knots, persistent=False)
        self.register_buffer("quad_weights", weights * target_cf, persistent=False)
        self.register_buffer("target_cf", target_cf, persistent=False)

    def _epps_pulley_2d(self, z_flat: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        # z_flat: [N_global, D] float32, directions: [D, M] float32
        slices = z_flat @ directions  # [N_global, M]
        angles = self.t_knots[:, None, None] * slices[None, :, :]  # [K, N_global, M]
        cf_real = torch.cos(angles).mean(dim=1)  # [K, M]
        cf_imag = torch.sin(angles).mean(dim=1)  # [K, M]
        diff_sq = (cf_real - self.target_cf[:, None]) ** 2 + cf_imag**2
        ep_per_slice = (diff_sq * self.quad_weights[:, None]).sum(dim=0)
        return ep_per_slice.mean()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate SIGReg in fp32 with differentiable all-gather across DDP ranks.

        Accepts ``[B, D]`` or ``[B, T, D]``. For ``[B, T, D]``, all-gathers across
        the batch dimension so each timestep is tested on ``B_global`` i.i.d. samples.
        """
        z_fp32 = z.float()
        z_global = _differentiable_all_gather(z_fp32)
        D = z_global.shape[-1]
        device = z_global.device

        with torch.no_grad():
            directions = torch.randn(D, self.num_slices, device=device, dtype=torch.float32)
            directions = nn.functional.normalize(directions, dim=0)
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                # dist.broadcast requires contiguous GPU/CPU tensor matching backend
                dist.broadcast(directions, src=0)

        if z_global.dim() == 2:
            return self._epps_pulley_2d(z_global, directions)
        if z_global.dim() == 3:
            T = z_global.shape[1]
            losses = [self._epps_pulley_2d(z_global[:, t, :], directions) for t in range(T)]
            return torch.stack(losses).mean()
        raise ValueError(f"Expected 2D or 3D tensor for SIGReg, got shape {tuple(z.shape)}")
