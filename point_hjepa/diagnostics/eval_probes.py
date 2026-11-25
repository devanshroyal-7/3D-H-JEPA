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

    z1_list, z2_list = [], []
    ee1_list, ach1_list = [], []
    ee2_list, ach2_list = [], []
    ply_written: list[Path] = []
    collected = 0
    s2 = raw_model.strides[0] if raw_model.num_levels >= 2 else 1

    for batch in val_loader:
        B = int(batch["batch_size"])
        T = int(batch["seq_len"])
        point = move_point_to_device(batch["point"], device)
        actions = batch["actions"].to(device, non_blocking=True)
        ach = batch["achieved_point"].to(device, non_blocking=True)
        ee = batch["ee_point"].to(device, non_blocking=True)

        out = raw_model(point, actions, batch_size=B, seq_len=T, return_attn=(len(ply_written) == 0))
        states = out["states"]

        z1 = states[0]  # [B, T, D]
        z1_list.append(z1.reshape(-1, z1.shape[-1]))
        ach1_list.append(ach.reshape(-1, 3))
        ee1_list.append(ee.reshape(-1, 3))

        if raw_model.num_levels >= 2:
            z2 = states[1]  # [B, T2, D]
            k_trans = (T - 1) // s2
            idx2 = torch.arange(0, (k_trans + 1) * s2, s2, device=device)
            z2_list.append(z2.reshape(-1, z2.shape[-1]))
            ach2_list.append(ach[:, idx2].reshape(-1, 3))
            ee2_list.append(ee[:, idx2].reshape(-1, 3))

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

    z1_all = torch.cat(z1_list, dim=0)[:max_frames]
    ach1_all = torch.cat(ach1_list, dim=0)[:max_frames]
    ee1_all = torch.cat(ee1_list, dim=0)[:max_frames]

    rank1, healthy1 = compute_effective_rank(z1_all)
    results: dict[str, float] = {
        "diag/eff_rank_l1": rank1,
        "diag/eff_rank_l1_healthy": float(healthy1),
    }

    split1 = max(2, int(0.8 * z1_all.shape[0]))
    if z1_all.shape[0] > split1 + 1:
        with torch.enable_grad():
            err_ach1 = fit_and_eval_coord_probe(
                z1_all[:split1], ach1_all[:split1], z1_all[split1:], ach1_all[split1:]
            )
            err_ee1 = fit_and_eval_coord_probe(
                z1_all[:split1], ee1_all[:split1], z1_all[split1:], ee1_all[split1:]
            )
        results["diag/probe_ach_l1_cm"] = err_ach1
        results["diag/probe_ee_l1_cm"] = err_ee1

    if z2_list:
        z2_all = torch.cat(z2_list, dim=0)[:max_frames]
        ach2_all = torch.cat(ach2_list, dim=0)[:max_frames]
        ee2_all = torch.cat(ee2_list, dim=0)[:max_frames]
        rank2, healthy2 = compute_effective_rank(z2_all)
        results["diag/eff_rank_l2"] = rank2
        results["diag/eff_rank_l2_healthy"] = float(healthy2)

        split2 = max(2, int(0.8 * z2_all.shape[0]))
        if z2_all.shape[0] > split2 + 1:
            with torch.enable_grad():
                err_ach2 = fit_and_eval_coord_probe(
                    z2_all[:split2], ach2_all[:split2], z2_all[split2:], ach2_all[split2:]
                )
                err_ee2 = fit_and_eval_coord_probe(
                    z2_all[:split2], ee2_all[:split2], z2_all[split2:], ee2_all[split2:]
                )
            results["diag/probe_ach_l2_cm"] = err_ach2
            results["diag/probe_ee_l2_cm"] = err_ee2
            if "diag/probe_ach_l1_cm" in results and err_ach2 > results["diag/probe_ach_l1_cm"] + 1.0:
                log.warning(
                    "L2 probe error (%.2f cm) degraded vs L1 (%.2f cm); consider scaling up gamma_2.",
                    err_ach2,
                    results["diag/probe_ach_l1_cm"],
                )

    raw_model.train(was_training)
    return results
