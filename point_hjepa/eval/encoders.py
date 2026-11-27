"""Frozen observation encoders for Point-H-JEPA, state-only, and JEPA4R baselines."""

from __future__ import annotations

import abc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import point_transformer_v3 as ptv3
import torch
from omegaconf import DictConfig, OmegaConf

from point_hjepa.models.point_hjepa import PointHJEPA, compute_min_l1_frames


def build_world_model_from_cfg(cfg: DictConfig) -> PointHJEPA:
    """Instantiate ``PointHJEPA`` from a saved Hydra training config."""
    pred_max_seq_len = cfg.predictor.get("max_seq_len", None)
    if pred_max_seq_len is None and cfg.get("num_l1_frames", None) is not None:
        min_l1 = compute_min_l1_frames(list(cfg.strides), list(cfg.context_lengths))
        pred_max_seq_len = max(64, min_l1, int(cfg.num_l1_frames))
    return PointHJEPA(
        num_levels=int(cfg.num_levels),
        strides=list(cfg.strides),
        context_lengths=list(cfg.context_lengths),
        action_dim=int(cfg.action_dim),
        frame_skip=int(cfg.frame_skip),
        latent_dim=int(cfg.latent_dim),
        action_l1_dim=int(cfg.action_l1_dim),
        macro_action_dim=int(cfg.macro_action_dim),
        action_pooler_type=str(cfg.action_pooler_type),
        backbone_preset=str(cfg.backbone.preset),
        backbone_pretrained=bool(cfg.backbone.pretrained),
        serialization_depth=int(cfg.backbone.serialization_depth),
        freeze_stages=int(cfg.backbone.freeze_stages),
        fourier_bands=int(cfg.backbone.fourier_bands),
        utonia_scale=float(cfg.backbone.scale),
        pooler_queries=int(cfg.pooler.num_queries),
        pooler_heads=int(cfg.pooler.num_heads),
        pooler_layers=int(cfg.pooler.num_layers),
        pooler_mlp_ratio=float(cfg.pooler.mlp_ratio),
        predictor_depth=int(cfg.predictor.depth),
        predictor_heads=int(cfg.predictor.num_heads),
        predictor_mlp_ratio=float(cfg.predictor.mlp_ratio),
        predictor_max_seq_len=None if pred_max_seq_len is None else int(pred_max_seq_len),
        lambda_sigreg=list(cfg.loss.lambda_sigreg),
        gamma_idm=list(cfg.loss.gamma_idm),
        sigreg_num_slices=int(cfg.loss.sigreg_num_slices),
        sigreg_t_min=float(cfg.loss.sigreg_t_min),
        sigreg_t_max=float(cfg.loss.sigreg_t_max),
        sigreg_num_knots=int(cfg.loss.sigreg_num_knots),
    )


def load_frozen_world_model(
    ckpt_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[PointHJEPA, DictConfig]:
    """Load a Stage-1 ``PointHJEPA`` checkpoint strictly frozen in eval mode."""
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Stage-1 Point-H-JEPA checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise KeyError(f"{path} has no 'config'; cannot reconstruct PointHJEPA")
    wm_cfg = OmegaConf.create(ckpt["config"])
    wm = build_world_model_from_cfg(wm_cfg)
    wm.load_state_dict(ckpt["model_state_dict"], strict=True)
    wm.to(device).eval().requires_grad_(False)
    return wm, wm_cfg


class FrozenObsEncoder(abc.ABC):
    """Common interface for frozen feature extractors in the OGBench harness."""

    name: str
    modality: str
    latent_dim: int

    @abc.abstractmethod
    def encode_batch(self, obs: Any) -> np.ndarray:
        """Encode a batch of observations into finite float32 ``[B, latent_dim]``."""

    def encode_one(self, obs: Any) -> np.ndarray:
        """Encode a single observation into finite float32 ``[latent_dim]``."""
        if isinstance(obs, dict):
            batched = {k: np.asarray(v)[None] for k, v in obs.items()}
        elif isinstance(obs, tuple):
            batched = tuple(np.asarray(v)[None] for v in obs)
        else:
            batched = np.asarray(obs)[None]
        return self.encode_batch(batched)[0]


class PointHJEPAEncoder(FrozenObsEncoder):
    """Frozen hierarchical encoder wrapping a Stage-1 ``PointHJEPA`` checkpoint."""

    VALID_MODES = ("l1", "l2", "l3", "concat", "concat_12")

    def __init__(
        self,
        ckpt_path: str | Path,
        *,
        device: str | torch.device | None = None,
        latent_mode: str = "concat",
        batch_size: int = 32,
    ) -> None:
        if latent_mode not in self.VALID_MODES:
            raise ValueError(f"latent_mode must be one of {self.VALID_MODES}, got {latent_mode!r}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.latent_mode = str(latent_mode)
        self.batch_size = max(1, int(batch_size))
        self.modality = "pcd"
        self.name = f"point_hjepa_{self.latent_mode}"

        self.model, self.wm_cfg = load_frozen_world_model(ckpt_path, device=self.device)
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("PointHJEPAEncoder failed frozen/eval invariant check")

        num_levels = int(self.model.num_levels)
        if self.latent_mode in ("l2", "concat_12") and num_levels < 2:
            raise ValueError(f"latent_mode={self.latent_mode!r} requires num_levels >= 2, got {num_levels}")
        if self.latent_mode == "l3" and num_levels < 3:
            raise ValueError(f"latent_mode='l3' requires num_levels >= 3, got {num_levels}")

        d_z = int(self.model.latent_dim)
        if self.latent_mode in ("l1", "l2", "l3"):
            self.latent_dim = d_z
        elif self.latent_mode == "concat_12":
            self.latent_dim = 2 * d_z
        else:
            self.latent_dim = num_levels * d_z

        self._preset = str(self.wm_cfg.backbone.preset)
        self._scale = float(self.wm_cfg.backbone.scale)
        self._center = bool(self.wm_cfg.backbone.center)

    @staticmethod
    def _unpack_clouds(obs: Any) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(obs, dict):
            xyz = np.asarray(obs["xyz"], dtype=np.float32)
            rgb = np.asarray(obs["rgb"], dtype=np.float32)
            return xyz, rgb
        if isinstance(obs, tuple) and len(obs) == 2:
            xyz = np.asarray(obs[0], dtype=np.float32)
            rgb = np.asarray(obs[1], dtype=np.float32)
            return xyz, rgb
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[-1] not in (3, 6):
            raise ValueError(f"Expected point cloud batch [B, N, 6] or [B, N, 3], got {arr.shape}")
        xyz = arr[:, :, :3]
        rgb = arr[:, :, 3:6] if arr.shape[-1] == 6 else np.zeros_like(xyz)
        return xyz, rgb

    def _select_latents(self, z1: torch.Tensor) -> torch.Tensor:
        levels = [z1]
        z_curr = z1
        for enc in self.model.state_encoders:
            z_curr = enc(z_curr)
            levels.append(z_curr)

        if self.latent_mode == "l1":
            return levels[0]
        if self.latent_mode == "l2":
            return levels[1]
        if self.latent_mode == "l3":
            return levels[2]
        if self.latent_mode == "concat_12":
            return torch.cat([levels[0], levels[1]], dim=-1)
        return torch.cat(levels, dim=-1)

    @torch.inference_mode()
    def encode_batch(self, obs: Any) -> np.ndarray:
        xyz, rgb = self._unpack_clouds(obs)
        total = int(xyz.shape[0])
        if total == 0:
            return np.zeros((0, self.latent_dim), dtype=np.float32)

        zeros = np.zeros((xyz.shape[1], 3), dtype=np.float32)
        outs: list[np.ndarray] = []
        for start in range(0, total, self.batch_size):
            end = min(total, start + self.batch_size)
            pts = [
                ptv3.prepare(
                    xyz[i],
                    rgb[i],
                    zeros,
                    preset=self._preset,
                    scale=self._scale,
                    center=self._center,
                    device="cpu",
                )[0]
                for i in range(start, end)
            ]
            collated = ptv3.collate(pts)
            for key in ("coord", "grid_coord", "feat", "offset", "batch"):
                if key in collated and isinstance(collated[key], torch.Tensor):
                    collated[key] = collated[key].to(self.device, non_blocking=True)
            z1 = self.model.encode_level1(collated)["z1"]
            feats = self._select_latents(z1).float().cpu().numpy()
            outs.append(feats)
        return np.ascontiguousarray(np.concatenate(outs, axis=0), dtype=np.float32)


class StateOnlyEncoder(FrozenObsEncoder):
    """Pass-through state encoder for privileged OGBench state-only baselines."""

    def __init__(self, state_dim: int) -> None:
        self.name = "state"
        self.modality = "state"
        self.latent_dim = int(state_dim)

    def encode_batch(self, obs: Any) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None]
        return np.ascontiguousarray(arr, dtype=np.float32)


class JEPA4RBaselineEncoder(FrozenObsEncoder):
    """Loads frozen JEPA4R paper baselines (``jepa4r``, ``lewm``, ``pldm``, ``lejepa``, ``visreg``, ``vjepa21``)."""

    METHODS = ("jepa4r", "lewm", "pldm", "lejepa", "visreg", "vjepa21")

    def __init__(
        self,
        method: str,
        task: str,
        *,
        ckpt_path: str | Path | None = None,
        jepa4r_root: str | Path = "third_party/JEPA4R",
        checkpoints_root: str | Path = "checkpoints/jepa4r",
        epoch: int = 10,
        device: str | torch.device | None = None,
        batch_size: int = 64,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f"Unknown JEPA4R baseline {method!r}; choose from {self.METHODS}")
        self.method = str(method)
        self.task = str(task)
        self.name = self.method
        self.modality = "pcd" if self.method == "jepa4r" else "rgb"
        self.batch_size = max(1, int(batch_size))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        jepa_dir = Path(jepa4r_root).resolve()
        if not jepa_dir.is_dir():
            raise FileNotFoundError(
                f"JEPA4R repo not found at {jepa_dir}. Run: python scripts/setup_jepa4r.py --skip-weights"
            )
        bc_dir = jepa_dir / "jepa4r" / "policy" / "bc"
        for p in (str(jepa_dir), str(bc_dir)):
            if p not in sys.path:
                sys.path.insert(0, p)

        resolved_ckpt = self._resolve_ckpt_path(Path(checkpoints_root), task, method, ckpt_path)
        self.model, dim, self.transform = self._load_model(method, resolved_ckpt, epoch)
        self.model = self.model.to(self.device).eval().requires_grad_(False)
        self.latent_dim = int(dim)

    @staticmethod
    def _resolve_ckpt_path(
        checkpoints_root: Path,
        task: str,
        method: str,
        ckpt_path: str | Path | None,
    ) -> Path:
        if ckpt_path is not None:
            p = Path(ckpt_path)
            if not p.exists():
                raise FileNotFoundError(f"Baseline checkpoint not found: {p}")
            return p

        candidates = [
            checkpoints_root / task / method / "world_model",
            checkpoints_root / task / method,
            checkpoints_root / method,
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        raise FileNotFoundError(
            f"Checkpoints for {method!r} on {task!r} not found under {checkpoints_root / task}. "
            f"Download them with: python scripts/setup_jepa4r.py --tasks {task}"
        )

    @staticmethod
    def _load_model(method: str, ckpt: Path, epoch: int) -> tuple[torch.nn.Module, int, Any]:
        import jepa_baselines
        import wm_embeddings

        if method == "jepa4r":
            model, dim, tf, _ = wm_embeddings.load_jepa4r(
                str(ckpt), epoch=epoch, center=(0.425, 0.0, 0.2), scale=4.0, use_rgb=True
            )
            return model, dim, tf
        if method == "lewm":
            model, dim, tf, _ = wm_embeddings.load_lewm(str(ckpt), epoch=epoch)
            return model, dim, tf
        if method == "pldm":
            model, dim, tf, _ = jepa_baselines.load_pldm(str(ckpt), epoch=epoch)
            return model, dim, tf
        if method == "lejepa":
            model, dim, tf, _ = jepa_baselines.load_lejepa(str(ckpt), epoch=epoch)
            return model, dim, tf
        if method == "visreg":
            ckpt_file = ckpt if ckpt.is_file() else ckpt / "visreg-vit-b-inet1k.pth"
            model, dim, tf, _ = jepa_baselines.load_visreg(str(ckpt_file))
            return model, dim, tf
        ckpt_file = ckpt if ckpt.is_file() else ckpt / "vjepa2_1_vitb_dist_vitG_384.pt"
        model, dim, tf, _ = jepa_baselines.load_vjepa21(str(ckpt_file))
        return model, dim, tf

    @torch.inference_mode()
    def encode_batch(self, obs: Any) -> np.ndarray:
        if self.modality == "pcd":
            arr = np.asarray(obs, dtype=np.float32)
            outs: list[np.ndarray] = []
            for start in range(0, len(arr), self.batch_size):
                batch_t = torch.stack([self.transform(p) for p in arr[start : start + self.batch_size]])
                outs.append(self.model(batch_t.to(self.device)).float().cpu().numpy())
            return np.ascontiguousarray(np.concatenate(outs, axis=0), dtype=np.float32)

        from PIL import Image

        arr = np.asarray(obs, dtype=np.uint8)
        outs = []
        for start in range(0, len(arr), self.batch_size):
            batch_t = torch.stack(
                [self.transform(Image.fromarray(img)) for img in arr[start : start + self.batch_size]]
            )
            outs.append(self.model(batch_t.to(self.device)).float().cpu().numpy())
        return np.ascontiguousarray(np.concatenate(outs, axis=0), dtype=np.float32)


def build_encoder(
    method: str,
    task: str,
    *,
    wm_ckpt: str | Path | None = None,
    latent_mode: str = "concat",
    state_dim: int | None = None,
    baseline_ckpt: str | Path | None = None,
    jepa4r_root: str | Path = "third_party/JEPA4R",
    checkpoints_root: str | Path = "checkpoints/jepa4r",
    device: str | torch.device | None = None,
) -> FrozenObsEncoder:
    """Create the appropriate ``FrozenObsEncoder`` for a benchmark method."""
    if method == "point_hjepa":
        if wm_ckpt is None:
            raise ValueError("method='point_hjepa' requires --wm-ckpt")
        return PointHJEPAEncoder(wm_ckpt, device=device, latent_mode=latent_mode)
    if method == "state":
        if state_dim is None:
            raise ValueError("method='state' requires state_dim")
        return StateOnlyEncoder(state_dim)
    return JEPA4RBaselineEncoder(
        method,
        task,
        ckpt_path=baseline_ckpt,
        jepa4r_root=jepa4r_root,
        checkpoints_root=checkpoints_root,
        device=device,
    )
