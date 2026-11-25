"""Hierarchical clip and hindsight goal datasets backed by Zarr v3 stores."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import point_transformer_v3 as ptv3
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from point_transformer_v3.structure import Point


def _normalize_sources(
    zarr_paths: str | Path | Sequence[str | Path | dict[str, Any]],
) -> list[tuple[Path, float]]:
    if isinstance(zarr_paths, (str, Path)):
        return [(Path(zarr_paths), 1.0)]
    sources: list[tuple[Path, float]] = []
    for item in zarr_paths:
        if isinstance(item, (str, Path)):
            sources.append((Path(item), 1.0))
        else:
            sources.append((Path(item["path"]), float(item.get("weight", 1.0))))
    return sources


def _split_episodes(
    episode_ends: np.ndarray,
    val_ratio: float,
    split: str,
    seed: int,
) -> np.ndarray:
    n_eps = len(episode_ends)
    if n_eps == 0:
        return np.zeros((0,), dtype=np.int64)
    if val_ratio <= 0.0 or n_eps == 1:
        return np.arange(n_eps, dtype=np.int64) if split == "train" else np.arange(n_eps, dtype=np.int64)

    perm = np.random.default_rng(seed).permutation(n_eps)
    n_val = max(1, min(n_eps - 1, int(round(n_eps * val_ratio))))
    if split == "val":
        return np.sort(perm[:n_val])
    if split == "train":
        return np.sort(perm[n_val:])
    raise ValueError(f"split must be 'train' or 'val', got {split!r}")


class HierarchicalClipDataset(Dataset):
    """Extracts episode-safe clips of ``num_l1_frames`` separated by ``frame_skip``.

    Each clip covers raw simulation span ``(num_l1_frames - 1) * frame_skip + 1``
    within a single episode (never crossing ``episode_ends``).
    """

    def __init__(
        self,
        zarr_paths: str | Path | Sequence[str | Path | dict[str, Any]],
        frame_skip: int = 3,
        num_l1_frames: int = 8,
        action_dim: int = 8,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.sources = _normalize_sources(zarr_paths)
        self.frame_skip = int(frame_skip)
        self.num_l1_frames = int(num_l1_frames)
        self.action_dim = int(action_dim)
        self.span = (self.num_l1_frames - 1) * self.frame_skip + 1

        self._groups: list[Any] | None = None
        self.clips: list[tuple[int, int]] = []
        self.sample_weights: list[float] = []

        for ds_idx, (path, weight) in enumerate(self.sources):
            import zarr

            root = zarr.open_group(str(path), mode="r")
            ep_ends = np.asarray(root["episode_ends"], dtype=np.int64)
            ep_starts = np.concatenate([np.zeros((1,), dtype=np.int64), ep_ends[:-1]])
            selected_eps = _split_episodes(ep_ends, val_ratio, split, seed + ds_idx)

            ds_clips = 0
            for ep_id in selected_eps:
                s = int(ep_starts[ep_id])
                e = int(ep_ends[ep_id])
                if e - s < self.span:
                    continue
                for start in range(s, e - self.span + 1):
                    self.clips.append((ds_idx, start))
                    ds_clips += 1

            if ds_clips > 0:
                per_clip_w = weight / float(ds_clips)
                self.sample_weights.extend([per_clip_w] * ds_clips)

        if not self.clips:
            raise RuntimeError(
                f"No valid clips of span {self.span} found for split={split!r} across {[str(p) for p, _ in self.sources]}"
            )

    def _ensure_open(self) -> list[Any]:
        if self._groups is None:
            import zarr

            self._groups = [zarr.open_group(str(p), mode="r") for p, _ in self.sources]
        return self._groups

    def make_weighted_sampler(self, num_samples: int | None = None) -> WeightedRandomSampler:
        n = len(self.sample_weights) if num_samples is None else num_samples
        weights = torch.as_tensor(self.sample_weights, dtype=torch.double)
        return WeightedRandomSampler(weights, num_samples=n, replacement=True)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        ds_idx, start = self.clips[index]
        root = self._ensure_open()[ds_idx]
        stop = start + self.span

        xyz_raw = np.asarray(root["xyz"][start:stop], dtype=np.float32)
        rgb_raw = np.asarray(root["rgb"][start:stop], dtype=np.float32)
        act_raw = np.asarray(root["action"][start:stop], dtype=np.float32)
        ach_raw = np.asarray(root["achieved_point"][start:stop], dtype=np.float32)
        ee_raw = np.asarray(root["ee_point"][start:stop], dtype=np.float32)

        frame_idx = np.arange(0, self.span, self.frame_skip, dtype=np.int64)
        xyz = xyz_raw[frame_idx]  # [T_l1, N, 3]
        rgb = rgb_raw[frame_idx]  # [T_l1, N, 3]
        achieved_point = ach_raw[frame_idx]  # [T_l1, 3]
        ee_point = ee_raw[frame_idx]  # [T_l1, 3]

        # Group frame_skip consecutive primitive actions per Level-1 transition: [T_l1 - 1, frame_skip * action_dim]
        act_blocks = [
            act_raw[i * self.frame_skip : (i + 1) * self.frame_skip].reshape(-1)
            for i in range(self.num_l1_frames - 1)
        ]
        actions = np.stack(act_blocks, axis=0).astype(np.float32)

        return {
            "xyz": xyz,
            "rgb": rgb,
            "actions": actions,
            "achieved_point": achieved_point,
            "ee_point": ee_point,
        }


class HindsightGoalDataset(Dataset):
    """Samples hindsight frame pairs ``(t, t')`` with ``t' > t`` within the same episode."""

    def __init__(
        self,
        zarr_paths: str | Path | Sequence[str | Path | dict[str, Any]],
        split: str = "train",
        val_ratio: float = 0.1,
        min_delta: int = 2,
        max_delta: int = 40,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.sources = _normalize_sources(zarr_paths)
        self.min_delta = int(min_delta)
        self.max_delta = int(max_delta)
        self.split = split
        self.rng = np.random.default_rng(seed + (100 if split == "val" else 0))

        self._groups: list[Any] | None = None
        self.samples: list[tuple[int, int, int]] = []  # (ds_idx, t, ep_end)

        for ds_idx, (path, _) in enumerate(self.sources):
            import zarr

            root = zarr.open_group(str(path), mode="r")
            ep_ends = np.asarray(root["episode_ends"], dtype=np.int64)
            ep_starts = np.concatenate([np.zeros((1,), dtype=np.int64), ep_ends[:-1]])
            selected_eps = _split_episodes(ep_ends, val_ratio, split, seed + ds_idx)
            for ep_id in selected_eps:
                s = int(ep_starts[ep_id])
                e = int(ep_ends[ep_id])
                for t in range(s, e - self.min_delta):
                    self.samples.append((ds_idx, t, e))

        if not self.samples:
            raise RuntimeError(f"No hindsight pairs found for split={split!r}")

    def _ensure_open(self) -> list[Any]:
        if self._groups is None:
            import zarr

            self._groups = [zarr.open_group(str(p), mode="r") for p, _ in self.sources]
        return self._groups

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        ds_idx, t, ep_end = self.samples[index]
        root = self._ensure_open()[ds_idx]

        hi = min(ep_end - 1, t + self.max_delta)
        lo = t + self.min_delta
        if self.split == "val":
            tp = (lo + hi) // 2
        else:
            tp = int(self.rng.integers(lo, hi + 1))

        return {
            "xyz_t": np.asarray(root["xyz"][t], dtype=np.float32),
            "rgb_t": np.asarray(root["rgb"][t], dtype=np.float32),
            "xyz_tp": np.asarray(root["xyz"][tp], dtype=np.float32),
            "rgb_tp": np.asarray(root["rgb"][tp], dtype=np.float32),
            "achieved_point_tp": np.asarray(root["achieved_point"][tp], dtype=np.float32),
        }


def make_ptv3_collate_fn(
    preset: str = "utonia_tiny",
    scale: float = 4.0,
    center: bool = False,
):
    """Build a collate_fn packing ``B * T`` frames into a single ragged ``Point`` batch."""

    def collate_fn(batch: list[dict[str, np.ndarray]]) -> dict[str, Any]:
        B = len(batch)
        T = int(batch[0]["xyz"].shape[0])
        points: list[Point] = []
        zeros_cache: np.ndarray | None = None

        for item in batch:
            xyz_seq = item["xyz"]
            rgb_seq = item["rgb"]
            if zeros_cache is None or zeros_cache.shape[0] != xyz_seq.shape[1]:
                zeros_cache = np.zeros((xyz_seq.shape[1], 3), dtype=np.float32)
            for t in range(T):
                point, _ = ptv3.prepare(
                    xyz_seq[t],
                    rgb_seq[t],
                    zeros_cache,
                    preset=preset,
                    scale=scale,
                    center=center,
                    device="cpu",
                )
                points.append(point)

        collated = ptv3.collate(points)
        return {
            "point": collated,
            "actions": torch.from_numpy(np.stack([item["actions"] for item in batch], axis=0)),
            "achieved_point": torch.from_numpy(
                np.stack([item["achieved_point"] for item in batch], axis=0)
            ),
            "ee_point": torch.from_numpy(np.stack([item["ee_point"] for item in batch], axis=0)),
            "batch_size": B,
            "seq_len": T,
        }

    return collate_fn


def make_goal_collate_fn(
    preset: str = "utonia_tiny",
    scale: float = 4.0,
    center: bool = False,
):
    """Collate hindsight ``(t, t')`` pairs into collated ``Point`` batches."""

    def collate_fn(batch: list[dict[str, np.ndarray]]) -> dict[str, Any]:
        pts_t: list[Point] = []
        pts_tp: list[Point] = []
        zeros_cache: np.ndarray | None = None

        for item in batch:
            if zeros_cache is None or zeros_cache.shape[0] != item["xyz_t"].shape[0]:
                zeros_cache = np.zeros((item["xyz_t"].shape[0], 3), dtype=np.float32)
            pt_t, _ = ptv3.prepare(
                item["xyz_t"],
                item["rgb_t"],
                zeros_cache,
                preset=preset,
                scale=scale,
                center=center,
                device="cpu",
            )
            pt_tp, _ = ptv3.prepare(
                item["xyz_tp"],
                item["rgb_tp"],
                zeros_cache,
                preset=preset,
                scale=scale,
                center=center,
                device="cpu",
            )
            pts_t.append(pt_t)
            pts_tp.append(pt_tp)

        return {
            "point_t": ptv3.collate(pts_t),
            "point_tp": ptv3.collate(pts_tp),
            "achieved_point_tp": torch.from_numpy(
                np.stack([item["achieved_point_tp"] for item in batch], axis=0)
            ),
            "batch_size": len(batch),
        }

    return collate_fn
