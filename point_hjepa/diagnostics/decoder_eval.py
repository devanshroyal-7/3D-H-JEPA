"""Episode-level encoding, imagination and point-cloud decoding utilities for the per-level decoders."""

from __future__ import annotations

from typing import Any

import numpy as np
import point_transformer_v3 as ptv3
import torch

from point_hjepa.diagnostics.eval_probes import move_point_to_device
from point_hjepa.models.pc_decoder.hierarchical_decoder import HierarchicalPointDecoder
from point_hjepa.models.pc_decoder.losses import nn_point_errors
from point_hjepa.models.point_hjepa import PointHJEPA


def episode_spans(root: Any) -> list[tuple[int, int]]:
    ends = np.asarray(root["episode_ends"][:], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def load_episode_l1(root: Any, start: int, end: int, frame_skip: int, action_dim: int) -> dict[str, np.ndarray]:
    """Level-1 view of one raw episode: frames at ``start + i * frame_skip`` and grouped primitive actions."""
    n_raw = end - start
    T1 = (n_raw - 1) // frame_skip + 1
    raw_idx = start + np.arange(T1) * frame_skip
    xyz = np.asarray(root["xyz"][start:end], dtype=np.float32)[raw_idx - start]
    rgb = np.asarray(root["rgb"][start:end], dtype=np.float32)[raw_idx - start]
    act = np.asarray(root["action"][start:end], dtype=np.float32)
    actions = np.stack(
        [act[i * frame_skip : (i + 1) * frame_skip].reshape(-1) for i in range(T1 - 1)], axis=0
    ).reshape(T1 - 1, frame_skip * action_dim)
    return {"xyz": xyz, "rgb": rgb, "actions": actions, "raw_idx": raw_idx}


@torch.no_grad()
def encode_episode(
    wm: PointHJEPA,
    ep: dict[str, np.ndarray],
    device: torch.device,
    preset: str = "utonia_tiny",
    scale: float = 4.0,
    center: bool = False,
    chunk: int = 64,
    amp_dtype: torch.dtype | None = torch.bfloat16,
) -> dict[str, Any]:
    """Encode every Level-1 frame of an episode and build all levels (``PointHJEPA.build_hierarchy``)."""
    xyz, rgb = ep["xyz"], ep["rgb"]
    zeros = np.zeros((xyz.shape[1], 3), dtype=np.float32)
    z_chunks = []
    use_amp = device.type == "cuda" and amp_dtype is not None
    for s in range(0, xyz.shape[0], chunk):
        pts = [
            ptv3.prepare(xyz[t], rgb[t], zeros, preset=preset, scale=scale, center=center, device="cpu")[0]
            for t in range(s, min(s + chunk, xyz.shape[0]))
        ]
        point = move_point_to_device(ptv3.collate(pts), device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            z_chunks.append(wm.encode_level1(point)["z1"].float())
    z1 = torch.cat(z_chunks, dim=0).unsqueeze(0)  # [1, T1, D]
    actions = torch.from_numpy(ep["actions"]).to(device).unsqueeze(0)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        hier = wm.build_hierarchy(z1, actions)
    hier["states"] = [s.float() for s in hier["states"]]
    hier["actions"] = [a.float() for a in hier["actions"]]
    return hier


@torch.no_grad()
def imagine_hierarchy(wm: PointHJEPA, hier: dict[str, Any], context: list[int] | None = None) -> dict[str, list]:
    """Per level: teacher-forced one-step predictions and an open-loop rollout from the first ``context[l]`` states.

    Returns ``one_step[l] [1, T_l - 1, D]`` (predicting states ``1..``) and ``rollout[l] [1, T_l - C_l, D]``
    (predicting states ``C_l..``), or ``None`` where the level is too short.
    """
    one_step, rollout, ctx_used = [], [], []
    for lvl in range(wm.num_levels):
        z, a = hier["states"][lvl], hier["actions"][lvl]
        C = int(context[lvl]) if context is not None else wm.context_lengths[lvl]
        one_step.append(wm.one_step_predictions(lvl, z, a).float() if z.shape[1] >= 2 else None)
        if z.shape[1] > C:
            rollout.append(wm.rollout(lvl, z[:, :C], a).float())
        else:
            rollout.append(None)
        ctx_used.append(C)
    return {"one_step": one_step, "rollout": rollout, "context": ctx_used}


@torch.no_grad()
def decode_latents(
    decoder: HierarchicalPointDecoder,
    z: torch.Tensor,
    level: int,
    num_points: int,
    chunk: int = 16,
    amp_dtype: torch.dtype | None = torch.bfloat16,
) -> torch.Tensor:
    """``z [N, D]`` -> world-frame clouds ``[N, num_points, 6]`` (fixed pixel subset), decoded in chunks."""
    outs = []
    use_amp = z.device.type == "cuda" and amp_dtype is not None
    for s in range(0, z.shape[0], chunk):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            outs.append(decoder.decode_points(z[s : s + chunk], level, num_points).float())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def cloud_errors_cm(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor, outlier_cm: float = 3.0) -> dict[str, torch.Tensor]:
    """Per-frame NN error stats in cm between ``pred [N, P, 3]`` and ``gt [N, M, 3]`` (world metres)."""
    errs = nn_point_errors(pred_xyz, gt_xyz) * 100.0  # [N, P + M]
    P = pred_xyz.shape[1]
    mean = 0.5 * (errs[:, :P].mean(-1) + errs[:, P:].mean(-1))
    return {
        "nn_cm": mean,
        "median_cm": errs.median(dim=-1).values,
        "p90_cm": errs.quantile(0.9, dim=-1),
        "outlier_frac": (errs > outlier_cm).float().mean(-1),
    }
