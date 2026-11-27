# Ported from JEPA4R (https://github.com/AIGeeksGroup/JEPA4R, jepa4r/decoder/losses.py), Apache-2.0.
"""Decoder loss (JEPA4R Eq. 2): ``L_dec = CD(P_hat, P) + SW_2^2(P_hat, P) + 0.1 * L_rgb(P_hat, P)``.

CD     symmetric Chamfer distance on xyz (squared distances)
SW_2^2 squared sliced Wasserstein-2 distance on xyz
L_rgb  L1 colour error of nearest-neighbour matched points
``nn_distance`` is the validation readout (mean two-sided NN distance, not squared).
"""

from __future__ import annotations

import torch


def _fp32(*tensors: torch.Tensor) -> list[torch.Tensor]:
    return [t.float() for t in tensors]


def _nn_index(a: torch.Tensor, b: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """argmin_j ||a_i - b_j|| over xyz for every i, without autograd memory. a (B, N, C), b (B, M, C) -> (B, N)."""
    with torch.no_grad():
        return torch.cat(
            [torch.cdist(a[:, s : s + chunk, :3], b[..., :3]).argmin(dim=-1) for s in range(0, a.size(1), chunk)],
            dim=1,
        )


def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def chamfer(
    pred: torch.Tensor, gt: torch.Tensor, color_weight: float = 0.1, chunk: int = 256
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Symmetric Chamfer on xyz (squared distances) + L1 colour on the matched pairs.

    The nearest-neighbour search runs under ``no_grad`` and only the matched pairs are differentiated: the argmin is
    locally constant, so this is the gradient of the min at O(B*N) instead of O(B*N*M) memory.
    """
    with torch.autocast(device_type=pred.device.type, enabled=False):
        pred, gt = _fp32(pred, gt)
        g_of_p = _gather(gt, _nn_index(pred, gt, chunk))  # nearest real point of every decoded point
        p_of_g = _gather(pred, _nn_index(gt, pred, chunk))  # nearest decoded point of every real point
        d_pg = (pred[..., :3] - g_of_p[..., :3]).pow(2).sum(-1).mean()
        d_gp = (gt[..., :3] - p_of_g[..., :3]).pow(2).sum(-1).mean()
        loss = d_pg + d_gp
        logs = {"cd": loss.detach()}
        if color_weight > 0 and pred.size(-1) > 3 and gt.size(-1) > 3:
            col = (pred[..., 3:] - g_of_p[..., 3:]).abs().mean() + (gt[..., 3:] - p_of_g[..., 3:]).abs().mean()
            loss = loss + color_weight * col
            logs["col"] = col.detach()
        return loss, logs


def sliced_wasserstein(pred: torch.Tensor, gt: torch.Tensor, n_proj: int = 100) -> torch.Tensor:
    """Squared sliced Wasserstein-2 distance between the xyz of two point sets.

    Both sets are projected on ``n_proj`` random directions and sorted; on a line the optimal one-to-one matching
    pairs equal ranks. Sets of different sizes are compared at the quantiles of the larger one.
    """
    with torch.autocast(device_type=pred.device.type, enabled=False):
        p, g = _fp32(pred[..., :3], gt[..., :3])
        theta = torch.nn.functional.normalize(torch.randn(n_proj, 3, device=p.device), dim=-1)
        sp = (p @ theta.T).sort(dim=1).values.transpose(1, 2)  # (B, n_proj, N)
        sg = (g @ theta.T).sort(dim=1).values.transpose(1, 2)  # (B, n_proj, M)
        if sp.size(-1) != sg.size(-1):
            n = max(sp.size(-1), sg.size(-1))

            def resample(s: torch.Tensor) -> torch.Tensor:
                return (
                    s
                    if s.size(-1) == n
                    else torch.nn.functional.interpolate(s, size=n, mode="linear", align_corners=True)
                )

            sp, sg = resample(sp), resample(sg)
        # x3: E[(theta . d)^2] = |d|^2 / 3, same units as Chamfer
        return 3.0 * (sp - sg).pow(2).mean()


@torch.no_grad()
def nn_distance(pred: torch.Tensor, gt: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """Mean of the two one-sided mean nearest-neighbour distances (not squared), per sample, in the input's units."""
    pred, gt = _fp32(pred, gt)
    a = (pred[..., :3] - _gather(gt[..., :3], _nn_index(pred, gt, chunk))).norm(dim=-1).mean(-1)
    b = (gt[..., :3] - _gather(pred[..., :3], _nn_index(gt, pred, chunk))).norm(dim=-1).mean(-1)
    return 0.5 * (a + b)


@torch.no_grad()
def nn_point_errors(pred: torch.Tensor, gt: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """Per-point two-sided NN distances ``(B, N + M)`` (pred->gt then gt->pred), for median / p90 / outlier stats."""
    pred, gt = _fp32(pred, gt)
    a = (pred[..., :3] - _gather(gt[..., :3], _nn_index(pred, gt, chunk))).norm(dim=-1)
    b = (gt[..., :3] - _gather(pred[..., :3], _nn_index(gt, pred, chunk))).norm(dim=-1)
    return torch.cat([a, b], dim=1)
