"""JEPA4R downstream encoder contract adapter and native runner helper."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from point_hjepa.eval.encoders import PointHJEPAEncoder

FAMILY = "point_hjepa"
DIM = 512


def adapter_dim(checkpoint_path: str | Path, latent_mode: str = "concat") -> int:
    """Inspect the output latent width of a Stage-1 checkpoint for ``latent_mode``."""
    ckpt = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    d_z = int(cfg["latent_dim"])
    num_levels = int(cfg["num_levels"])
    if latent_mode in ("l1", "l2", "l3"):
        return d_z
    if latent_mode == "concat_12":
        return 2 * d_z
    return num_levels * d_z


def load_frozen(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
    latent_mode: str = "concat",
) -> PointHJEPAEncoder:
    """JEPA4R encoder-adapter entrypoint: load a frozen ``PointHJEPAEncoder``."""
    return PointHJEPAEncoder(checkpoint_path, device=device, latent_mode=latent_mode)


def encode_pcd(
    encoder: PointHJEPAEncoder,
    xyzrgb: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Encode ``[N, 6]`` or ``[B, N, 6]`` XYZ-RGB point clouds into float32 latents."""
    arr = xyzrgb.detach().cpu().numpy() if isinstance(xyzrgb, torch.Tensor) else np.asarray(xyzrgb)
    if arr.ndim == 2:
        return encoder.encode_one(arr)
    return encoder.encode_batch(arr)


def build_jepa4r_native_command(
    task: str,
    method: str,
    *,
    jepa4r_root: str | Path = "third_party/JEPA4R",
    tag: str = "point_hjepa_eval",
    extra_overrides: list[str] | None = None,
) -> list[str]:
    """Construct a command to invoke JEPA4R's native ``train_policy`` runner."""
    root = Path(jepa4r_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"JEPA4R repository not found at {root}. Run: python scripts/setup_jepa4r.py --skip-weights"
        )
    cmd = [
        sys.executable,
        "-m",
        "jepa4r",
        "train_policy",
        f"env={task}",
        f"method={method}",
        f"tag={tag}",
    ]
    if extra_overrides:
        cmd.extend(extra_overrides)
    return cmd
