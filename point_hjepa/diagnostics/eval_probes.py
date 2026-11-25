"""Epoch diagnostics: effective rank (singular value entropy) and frozen 3D coord probes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from point_hjepa.diagnostics.export_attn_ply import export_batch_attention_plys

log = logging.getLogger(__name__)


def compute_effective_rank(z: torch.Tensor) -> tuple[float, bool]:
    """Compute Roy-Vetterli effective rank ``exp(H(sigma_norm))`` on ``Z [N, D]``.

    Returns ``(effective_rank, is_healthy)`` where ``is_healthy`` requires
    ``effective_rank >= 0.5 * D``.
    """
    z_c = z.detach().float()
    z_c = z_c - z_c.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(z_c)
    s_sum = s.sum()
    if s_sum <= 1e-12:
        return 0.0, False
    p = (s / s_sum).clamp_min(1e-12)
    entropy = -(p * torch.log(p)).sum()
    eff_rank = float(torch.exp(entropy).item())
    D = z_c.shape[1]
    healthy = eff_rank >= 0.5 * D
    if not healthy:
        log.warning("Effective rank %.2f fell below 0.5 * D (%.1f)!", eff_rank, 0.5 * D)
    return eff_rank, healthy


def fit_and_eval_coord_probe(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    hidden_dim: int = 128,
    steps: int = 250,
    lr: float = 1e-2,
) -> float:
    """Fit a 2-layer MLP probe from frozen latents ``Z`` to 3D coords and return L2 error (cm)."""
    device = z_train.device
    z_tr = z_train.detach().float()
    y_tr = y_train.detach().float()
    z_va = z_val.detach().float()
    y_va = y_val.detach().float()

    probe = nn.Sequential(
        nn.Linear(z_tr.shape[1], hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 3),
    ).to(device)

    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    probe.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(probe(z_tr), y_tr)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        err_m = torch.linalg.norm(probe(z_va) - y_va, dim=-1).mean()
    return float(err_m.item() * 100.0)


def move_point_to_device(point: Any, device: torch.device) -> Any:
    for k in ("coord", "grid_coord", "feat", "offset", "batch"):
        if k in point and isinstance(point[k], torch.Tensor):
            point[k] = point[k].to(device, non_blocking=True)
    return point


@torch.no_grad()
def run_epoch_diagnostics(
    model: nn.Module,
    val_loader: Any,
    device: torch.device,
    output_dir: str | Path,
    epoch: int,
    max_frames: int = 1024,
    ply_frames: int = 4,
    utonia_scale: float = 4.0,
) -> dict[str, float]:
    """Evaluate effective rank, frozen 3D coordinate probes, and attention PLY export."""
    raw_model = model.module if hasattr(model, "module") else model
    was_training = raw_model.training
    raw_model.eval()

    num_levels = int(raw_model.num_levels)
    z_lists: list[list[torch.Tensor]] = [[] for _ in range(num_levels)]
    ach_lists: list[list[torch.Tensor]] = [[] for _ in range(num_levels)]
    ee_lists: list[list[torch.Tensor]] = [[] for _ in range(num_levels)]
    val_metric_sums: dict[str, float] = {}
    val_metric_count = 0
    ply_written: list[Path] = []
    collected = 0

    for batch in val_loader:
        B = int(batch["batch_size"])
        T = int(batch["seq_len"])
        point = move_point_to_device(batch["point"], device)
        actions = batch["actions"].to(device, non_blocking=True)
        ach = batch["achieved_point"].to(device, non_blocking=True)
        ee = batch["ee_point"].to(device, non_blocking=True)

        out = raw_model(point, actions, batch_size=B, seq_len=T, return_attn=(len(ply_written) == 0))
        states = out["states"]
        if "metrics" in out:
            for k, v in out["metrics"].items():
                val = float(v.detach().item()) if isinstance(v, torch.Tensor) else float(v)
                val_metric_sums[k] = val_metric_sums.get(k, 0.0) + val * B
            val_metric_count += B

        idx = torch.arange(T, device=device, dtype=torch.long)
        for lvl in range(num_levels):
            if lvl > 0:
                stride = int(raw_model.strides[lvl - 1])
                k_trans = (idx.numel() - 1) // stride
                step_idx = torch.arange(0, (k_trans + 1) * stride, stride, device=device, dtype=torch.long)
                idx = idx[step_idx]
            z_l = states[lvl]
            z_lists[lvl].append(z_l.reshape(-1, z_l.shape[-1]))
            ach_lists[lvl].append(ach[:, idx].reshape(-1, 3))
            ee_lists[lvl].append(ee[:, idx].reshape(-1, 3))

        if not ply_written and ply_frames > 0:
            ply_written = export_batch_attention_plys(
                out["coord"],
                out["offset"],
                out["attn_weights"],
                Path(output_dir) / "attn_ply",
                epoch=epoch,
                max_frames=ply_frames,
                utonia_scale=utonia_scale,
            )

        collected += B * T
        if collected >= max_frames:
            break

    results: dict[str, float] = {}
    if val_metric_count > 0:
        for k, total in val_metric_sums.items():
            results[f"val/{k}"] = total / val_metric_count
    for lvl in range(num_levels):
        if not z_lists[lvl]:
            continue
        lnum = lvl + 1
        z_all = torch.cat(z_lists[lvl], dim=0)[:max_frames]
        ach_all = torch.cat(ach_lists[lvl], dim=0)[:max_frames]
        ee_all = torch.cat(ee_lists[lvl], dim=0)[:max_frames]

        rank_l, healthy_l = compute_effective_rank(z_all)
        results[f"diag/eff_rank_l{lnum}"] = rank_l
        results[f"diag/eff_rank_l{lnum}_healthy"] = float(healthy_l)

        split = max(2, int(0.8 * z_all.shape[0]))
        if z_all.shape[0] > split + 1:
            with torch.enable_grad():
                err_ach = fit_and_eval_coord_probe(
                    z_all[:split], ach_all[:split], z_all[split:], ach_all[split:]
                )
                err_ee = fit_and_eval_coord_probe(
                    z_all[:split], ee_all[:split], z_all[split:], ee_all[split:]
                )
            results[f"diag/probe_ach_l{lnum}_cm"] = err_ach
            results[f"diag/probe_ee_l{lnum}_cm"] = err_ee
            if lvl > 0 and "diag/probe_ach_l1_cm" in results and err_ach > results["diag/probe_ach_l1_cm"] + 1.0:
                log.warning(
                    "L%d probe error (%.2f cm) degraded vs L1 (%.2f cm); consider scaling up gamma_%d.",
                    lnum,
                    err_ach,
                    results["diag/probe_ach_l1_cm"],
                    lnum,
                )

    raw_model.train(was_training)
    return results
