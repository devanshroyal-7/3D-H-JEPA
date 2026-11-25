"""Export Attentive Pooling Bridge attention weights onto 3D point clouds (.ply)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh


def _blue_to_red_rgba(weights_01: np.ndarray) -> np.ndarray:
    """Map normalized scalar weights in [0, 1] to RGBA uint8 (blue -> cyan -> yellow -> red)."""
    w = np.clip(weights_01, 0.0, 1.0).astype(np.float32)
    r = np.clip(2.0 * w - 0.5, 0.0, 1.0)
    g = np.clip(1.0 - np.abs(2.0 * w - 1.0) * 0.5, 0.0, 1.0) * (1.0 - np.abs(2.0 * w - 1.0))
    # Clean monotonic blue (0,0,255) -> purple/yellow -> red (255,0,0)
    r = w
    g = 0.25 * (1.0 - np.abs(2.0 * w - 1.0))
    b = 1.0 - w
    rgb = np.stack([r, g, b], axis=-1)
    rgb_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    alpha = np.full((w.shape[0], 1), 255, dtype=np.uint8)
    return np.concatenate([rgb_u8, alpha], axis=-1)


def export_attention_ply(
    coord: torch.Tensor | np.ndarray,
    attn_weights: torch.Tensor | np.ndarray,
    out_path: str | Path,
    utonia_scale: float = 4.0,
) -> Path:
    """Save a single frame's ``out.coord`` + scalar bridge attention as a colored ``.ply``.

    ``coord`` is divided by ``utonia_scale`` to restore metric robot-base coordinates.
    """
    pts = coord.detach().cpu().numpy() if isinstance(coord, torch.Tensor) else np.asarray(coord)
    w = (
        attn_weights.detach().cpu().numpy()
        if isinstance(attn_weights, torch.Tensor)
        else np.asarray(attn_weights)
    )
    pts = pts.astype(np.float32) / max(float(utonia_scale), 1e-6)
    w = w[: pts.shape[0]].astype(np.float32)
    w_min, w_max = float(w.min()), float(w.max())
    w_norm = (w - w_min) / max(w_max - w_min, 1e-8)

    rgba = _blue_to_red_rgba(w_norm)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(vertices=pts, colors=rgba).export(str(out))
    return out


def export_batch_attention_plys(
    coord: torch.Tensor,
    offset: torch.Tensor,
    attn_weights: torch.Tensor,
    out_dir: str | Path,
    epoch: int,
    max_frames: int = 4,
    utonia_scale: float = 4.0,
) -> list[Path]:
    """Slice ``max_frames`` individual frames from ragged ``coord/offset`` and export ``.ply``."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ends = offset.detach().cpu().tolist()
    starts = [0, *ends[:-1]]
    written: list[Path] = []
    for idx in range(min(max_frames, len(ends))):
        s, e = starts[idx], ends[idx]
        if e <= s:
            continue
        p = export_attention_ply(
            coord[s:e],
            attn_weights[idx, : e - s],
            out_path / f"epoch_{epoch:03d}_frame_{idx}.ply",
            utonia_scale=utonia_scale,
        )
        written.append(p)
    return written
