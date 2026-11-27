"""Construct / save / load ``HierarchicalPointDecoder`` from config dicts and checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from point_hjepa.models.pc_decoder.hierarchical_decoder import HierarchicalPointDecoder


def estimate_workspace_center(zarr_paths: Sequence[str | Path], max_frames: int = 256, seed: int = 0) -> list[float]:
    """Midpoint of the 1st / 99th percentile of xyz over up to ``max_frames`` frames spread across the stores."""
    import zarr

    rng = np.random.default_rng(seed)
    per_store = max(1, max_frames // max(1, len(zarr_paths)))
    chunks = []
    for p in zarr_paths:
        root = zarr.open_group(str(p), mode="r")
        n = int(root["xyz"].shape[0])
        idx = np.sort(rng.choice(n, size=min(n, per_store), replace=False))
        for i in idx:
            pts = np.asarray(root["xyz"][int(i)], dtype=np.float32)
            chunks.append(pts[rng.choice(pts.shape[0], size=min(512, pts.shape[0]), replace=False)])
    x = np.concatenate(chunks, axis=0)
    x = x[np.isfinite(x).all(axis=1)]
    lo, hi = np.percentile(x, 1, axis=0), np.percentile(x, 99, axis=0)
    return [float(v) for v in 0.5 * (lo + hi)]


def decoder_kwargs(dcfg: Mapping[str, Any], num_levels: int, latent_dim: int, target_center: Sequence[float]) -> dict:
    """``HierarchicalPointDecoder`` constructor kwargs (minus the trunk) from the ``decoder`` config block."""
    return dict(
        num_levels=int(num_levels),
        latent_dim=int(latent_dim),
        gen_dim=int(dcfg["gen_dim"]),
        gen_hidden=int(dcfg["gen_hidden"]),
        gen_depth=int(dcfg["gen_depth"]),
        query_dim=int(dcfg["query_dim"]),
        color=bool(dcfg.get("rgb_branch", True)),
        color_weight=float(dcfg["loss"]["rgb_l1"]),
        sw_weight=float(dcfg["loss"]["sliced_w2"]),
        sw_projections=int(dcfg["loss"]["sw_projections"]),
        target_center=[float(c) for c in target_center],
        target_scale=float(dcfg["target_scale"]),
        view=[float(v) for v in dcfg["view"]],
    )


def build_trunk(dcfg: Mapping[str, Any]) -> nn.Module:
    from point_hjepa.models.pc_decoder.page4d_trunk import PAGE4DTrunk

    return PAGE4DTrunk(
        weights=dcfg.get("page4d_ckpt"),
        page4d_root=dcfg.get("page4d_root"),
        grid=int(dcfg["grid"]),
        grad_checkpointing=bool(dcfg.get("grad_checkpointing", True)),
    )


def save_decoder_checkpoint(path: str | Path, decoder: HierarchicalPointDecoder, meta: Mapping[str, Any]) -> None:
    """Trainable state only (heads + LoRA banks + target frame): MBs, not the 6.8 GB trunk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"decoder_state_dict": decoder.trainable_state_dict(), **dict(meta)}, path)


def load_decoder_checkpoint(
    path: str | Path,
    trunk: nn.Module | None = None,
    device: str | torch.device = "cpu",
) -> tuple[HierarchicalPointDecoder, dict[str, Any]]:
    """Rebuild a decoder from ``save_decoder_checkpoint`` output (the trunk is loaded from PAGE-4D's weights unless
    given). Returns ``(decoder, checkpoint_dict)``."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    kw = ckpt["decoder_kwargs"]
    dcfg = ckpt["decoder_config"]
    decoder = HierarchicalPointDecoder(trunk if trunk is not None else build_trunk(dcfg), **kw)
    if ckpt.get("lora_rank"):
        decoder.enable_lora(int(ckpt["lora_rank"]))
    decoder.load_trainable_state_dict(ckpt["decoder_state_dict"])
    return decoder.to(device), ckpt
