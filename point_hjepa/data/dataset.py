"""Window sampler for point-cloud episodes.

A window is anchored at the last context frame ``t`` inside a single episode.
Level 1 uses ``frameskip`` (default 5). Level 2 uses ``level2_stride``
(default 10). Both horizons have to lie entirely inside that episode; samples
that would cross ``episode_ends`` are not indexed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def level_indices(anchor: int, context: int, stride: int, horizon: int) -> tuple[list[int], list[int]]:
    context_idx = [anchor - (context - 1 - i) * stride for i in range(context)]
    future_idx = [anchor + (j + 1) * stride for j in range(horizon)]
    return context_idx, future_idx


def window_fits(length: int, anchor: int, context: int, stride: int, horizon: int) -> bool:
    context_idx, future_idx = level_indices(anchor, context, stride, horizon)
    indices = context_idx + future_idx
    return min(indices) >= 0 and max(indices) < length


def valid_anchors(
    length: int,
    *,
    context_frames: int,
    frameskip: int,
    horizon: int,
    level2_stride: int,
    level2_horizon: int,
) -> list[int]:
    anchors = []
    for anchor in range(length):
        level1 = window_fits(length, anchor, context_frames, frameskip, horizon)
        level2 = window_fits(length, anchor, context_frames, level2_stride, level2_horizon)
        if level1 and level2:
            anchors.append(anchor)
    return anchors


def actions_between(actions: np.ndarray, starts: list[int], stride: int) -> np.ndarray:
    """Stack the `stride` actions that leave each frame in `starts` except the last."""
    chunks = [actions[start : start + stride] for start in starts[:-1]]
    return np.stack(chunks, axis=0)


class PointCloudWindowDataset(torch.utils.data.Dataset):
    """Random-access windows over a Zarr or HDF5 episode dataset.

    Each item is a dict of torch tensors:

    - ``context_points``: ``(context, N, 3)`` at stride ``frameskip``
    - ``context_actions``: ``(context - 1, frameskip, action_dim)``
    - ``target_points``: ``(horizon, N, 3)``
    - ``target_actions``: ``(horizon, frameskip, action_dim)``
    - ``level2_context_points`` / ``level2_target_points`` at ``level2_stride``
    - ``level2_context_actions`` / ``level2_target_actions``
    - ``goal_coords`` at the anchor frame
    - ``episode_index``
    """

    def __init__(
        self,
        path: str | Path,
        *,
        context_frames: int = 3,
        frameskip: int = 5,
        horizon: int = 4,
        level2_stride: int = 10,
        level2_horizon: int = 2,
    ) -> None:
        self.path = Path(path)
        self.context_frames = int(context_frames)
        self.frameskip = int(frameskip)
        self.horizon = int(horizon)
        self.level2_stride = int(level2_stride)
        self.level2_horizon = int(level2_horizon)
        self._root, self._close = _open_readonly(self.path)
        self.points = self._root["points"]
        self.actions = self._root["actions"]
        self.goal = self._root["goal_coords"]
        self.ends = np.asarray(self._root["episode_ends"][:], dtype=np.int64)
        self._index: list[tuple[int, int]] = []
        start = 0
        for episode, end in enumerate(self.ends.tolist()):
            length = int(end) - start
            for anchor in valid_anchors(
                length,
                context_frames=self.context_frames,
                frameskip=self.frameskip,
                horizon=self.horizon,
                level2_stride=self.level2_stride,
                level2_horizon=self.level2_horizon,
            ):
                self._index.append((episode, anchor))
            start = int(end)

    def __len__(self) -> int:
        return len(self._index)

    def episode_span(self, episode: int) -> tuple[int, int]:
        start = 0 if episode == 0 else int(self.ends[episode - 1])
        return start, int(self.ends[episode])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode, anchor = self._index[index]
        start, end = self.episode_span(episode)
        points = np.asarray(self.points[start:end])
        actions = np.asarray(self.actions[start:end])
        goal = np.asarray(self.goal[start:end])
        sample = _pack_window(
            points,
            actions,
            goal[anchor],
            anchor,
            context_frames=self.context_frames,
            frameskip=self.frameskip,
            horizon=self.horizon,
            level2_stride=self.level2_stride,
            level2_horizon=self.level2_horizon,
        )
        sample["episode_index"] = torch.tensor(episode, dtype=torch.long)
        return sample

    def close(self) -> None:
        if self._close is not None:
            self._close()
            self._close = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _pack_window(points, actions, goal, anchor, *, context_frames, frameskip, horizon, level2_stride, level2_horizon):
    ctx1, fut1 = level_indices(anchor, context_frames, frameskip, horizon)
    ctx2, fut2 = level_indices(anchor, context_frames, level2_stride, level2_horizon)
    # The action stored at frame i is the command applied from that frame.
    # A stride-k jump consumes actions[i : i+k], which stops before the next frame.
    return {
        "context_points": torch.from_numpy(np.asarray(points[ctx1], dtype=np.float32)),
        "context_actions": torch.from_numpy(actions_between(actions, ctx1, frameskip).astype(np.float32)),
        "target_points": torch.from_numpy(np.asarray(points[fut1], dtype=np.float32)),
        "target_actions": torch.from_numpy(actions_between(actions, [ctx1[-1], *fut1], frameskip).astype(np.float32)),
        "level2_context_points": torch.from_numpy(np.asarray(points[ctx2], dtype=np.float32)),
        "level2_context_actions": torch.from_numpy(actions_between(actions, ctx2, level2_stride).astype(np.float32)),
        "level2_target_points": torch.from_numpy(np.asarray(points[fut2], dtype=np.float32)),
        "level2_target_actions": torch.from_numpy(
            actions_between(actions, [ctx2[-1], *fut2], level2_stride).astype(np.float32)
        ),
        "goal_coords": torch.from_numpy(np.asarray(goal, dtype=np.float32)),
    }


def _open_readonly(path: Path):
    if path.suffix in (".h5", ".hdf5") or path.is_file():
        import h5py

        handle = h5py.File(path, "r")
        return handle, handle.close
    import zarr

    root = zarr.open_group(path, mode="r")
    return root, None
