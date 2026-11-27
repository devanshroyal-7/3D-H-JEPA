"""Shared Zarr v3 chunked trajectory writer for 6D RGB-XYZ point-cloud datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr


class ZarrTrajectoryWriter:
    """Append chunked episodes to a Zarr v3 directory store with zstd compression."""

    def __init__(
        self,
        path: str | Path,
        num_points: int = 4096,
        action_dim: int = 8,
        proprio_dim: int = 26,
        chunk_steps: int = 256,
        attrs: dict[str, Any] | None = None,
        xyz_dtype: str = "float32",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.codec = zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
        self.root = zarr.open_group(str(self.path), mode="w")

        self.xyz = self.root.create_array(
            "xyz",
            shape=(0, num_points, 3),
            chunks=(chunk_steps, num_points, 3),
            dtype=xyz_dtype,
            compressors=self.codec,
        )
        self.rgb = self.root.create_array(
            "rgb",
            shape=(0, num_points, 3),
            chunks=(chunk_steps, num_points, 3),
            dtype="uint8",
            compressors=self.codec,
        )
        self.action = self.root.create_array(
            "action",
            shape=(0, action_dim),
            chunks=(chunk_steps * 4, action_dim),
            dtype="float32",
            compressors=self.codec,
        )
        self.proprio = self.root.create_array(
            "proprio",
            shape=(0, proprio_dim),
            chunks=(chunk_steps * 4, proprio_dim),
            dtype="float32",
            compressors=self.codec,
        )
        self.achieved_point = self.root.create_array(
            "achieved_point",
            shape=(0, 3),
            chunks=(chunk_steps * 4, 3),
            dtype="float32",
            compressors=self.codec,
        )
        self.ee_point = self.root.create_array(
            "ee_point",
            shape=(0, 3),
            chunks=(chunk_steps * 4, 3),
            dtype="float32",
            compressors=self.codec,
        )
        self.episode_ends = self.root.create_array(
            "episode_ends",
            shape=(0,),
            chunks=(1024,),
            dtype="int64",
            compressors=self.codec,
        )
        if attrs:
            for k, v in attrs.items():
                self.root.attrs[k] = v
        self.total_steps = 0
        self.num_episodes = 0

    def append_episode(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        action: np.ndarray,
        proprio: np.ndarray,
        achieved_point: np.ndarray,
        ee_point: np.ndarray,
    ) -> None:
        T = int(xyz.shape[0])
        if T == 0:
            return
        assert rgb.shape[:2] == (T, xyz.shape[1])
        assert action.shape[0] == T
        assert proprio.shape[0] == T
        assert achieved_point.shape == (T, 3)
        assert ee_point.shape == (T, 3)

        self.xyz.append(xyz.astype(self.xyz.dtype, copy=False))
        self.rgb.append(rgb.astype(np.uint8, copy=False))
        self.action.append(action.astype(np.float32, copy=False))
        self.proprio.append(proprio.astype(np.float32, copy=False))
        self.achieved_point.append(achieved_point.astype(np.float32, copy=False))
        self.ee_point.append(ee_point.astype(np.float32, copy=False))

        self.total_steps += T
        self.num_episodes += 1
        self.episode_ends.append(np.array([self.total_steps], dtype=np.int64))


ZarrEpisodeWriter = ZarrTrajectoryWriter
