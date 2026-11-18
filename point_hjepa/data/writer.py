"""Asynchronous chunked episode writer (Zarr, with an HDF5 option)."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import numpy as np


class AsyncEpisodeWriter:
    """Append whole episodes from a background thread.

    The main thread only enqueues episode dictionaries. Arrays are resized in
    chunks as episodes arrive. ``episode_ends`` stores the exclusive index of
    each episode in the frame stream.
    """

    def __init__(self, path: str | Path, fmt: str = "zarr") -> None:
        self.path = Path(path)
        self.fmt = fmt.lower()
        if self.fmt not in ("zarr", "hdf5", "h5"):
            raise ValueError(f"Unsupported dataset format {fmt!r}")
        if self.fmt == "h5":
            self.fmt = "hdf5"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue = queue.Queue(maxsize=4)
        self._error: BaseException | None = None
        self._frames = 0
        self._episodes = 0
        self._attrs: dict = {}
        self._thread = threading.Thread(target=self._loop, name="episode-writer", daemon=True)
        self._thread.start()

    def set_attrs(self, attrs: dict) -> None:
        self._attrs = attrs

    def write_episode(self, episode: dict) -> None:
        if self._error is not None:
            raise RuntimeError("dataset writer failed") from self._error
        # The caller may reuse buffers; copy before the thread reads them.
        copied = {key: (np.array(value) if isinstance(value, np.ndarray) else value) for key, value in episode.items()}
        self._queue.put(copied)

    def close(self) -> dict:
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("dataset writer failed") from self._error
        return {"frames": self._frames, "episodes": self._episodes, "path": str(self.path)}

    def _loop(self) -> None:
        handle = None
        try:
            while True:
                episode = self._queue.get()
                if episode is None:
                    break
                if handle is None:
                    handle = self._open_new(episode)
                self._append(handle, episode)
        except Exception as exc:  # noqa: BLE001 - surfaced from close()
            self._error = exc
        finally:
            if handle is not None:
                self._finalize(handle)

    def _open_new(self, episode: dict):
        points = episode["points"]
        frames, npoints, _ = points.shape
        goal_dim = int(np.asarray(episode["goal_coords"]).shape[-1])
        state_dim = int(np.asarray(episode["robot_state"]).shape[-1])
        action_dim = int(np.asarray(episode["actions"]).shape[-1])
        spec = {
            "points": ((0, npoints, 3), np.float32),
            "actions": ((0, action_dim), np.float32),
            "goal_coords": ((0, goal_dim), np.float32),
            "robot_state": ((0, state_dim), np.float32),
            "point_count": ((0,), np.int32),
            "success": ((0,), np.bool_),
            "done": ((0,), np.bool_),
            "episode_ends": ((0,), np.int64),
            "episode_source": ((0,), np.int8),
            "episode_success": ((0,), np.bool_),
            "episode_plan_failures": ((0,), np.int32),
        }
        if "rgb" in episode:
            spec["rgb"] = ((0, npoints, 3), np.uint8)
        if self.fmt == "zarr":
            import zarr

            root = zarr.open_group(self.path, mode="w")
            for name, (shape, dtype) in spec.items():
                chunks = (min(64, max(1, frames)),) + shape[1:]
                root.create_array(name, shape=shape, chunks=chunks, dtype=dtype)
            return root
        import h5py

        root = h5py.File(self.path, "w")
        for name, (shape, dtype) in spec.items():
            maxshape = (None,) + shape[1:]
            root.create_dataset(name, shape=shape, maxshape=maxshape, dtype=dtype, chunks=True)
        return root

    def _append(self, root, episode: dict) -> None:
        points = np.asarray(episode["points"])
        count = points.shape[0]
        start = self._frames
        end = start + count
        mapping = {
            "points": points.astype(np.float32, copy=False),
            "actions": np.asarray(episode["actions"], dtype=np.float32),
            "goal_coords": np.asarray(episode["goal_coords"], dtype=np.float32),
            "robot_state": np.asarray(episode["robot_state"], dtype=np.float32),
            "point_count": np.asarray(episode["point_count"], dtype=np.int32),
            "success": np.asarray(episode["success"], dtype=np.bool_),
            "done": np.asarray(episode["done"], dtype=np.bool_),
        }
        if "rgb" in episode:
            mapping["rgb"] = np.asarray(episode["rgb"], dtype=np.uint8)
        for name, value in mapping.items():
            self._extend(root[name], value)
        self._extend(root["episode_ends"], np.array([end], dtype=np.int64))
        self._extend(root["episode_source"], np.array([int(episode["source"])], dtype=np.int8))
        self._extend(root["episode_success"], np.array([bool(episode["episode_success"])], dtype=np.bool_))
        self._extend(
            root["episode_plan_failures"],
            np.array([int(episode.get("plan_failures", 0))], dtype=np.int32),
        )
        self._frames = end
        self._episodes += 1

    def _extend(self, array, value: np.ndarray) -> None:
        value = np.asarray(value)
        if value.ndim == 0:
            value = value.reshape(1)
        old = int(array.shape[0])
        new = old + int(value.shape[0])
        array.resize((new, *array.shape[1:]))
        array[old:new] = value

    def _finalize(self, root) -> None:
        payload = dict(self._attrs)
        payload["num_frames"] = int(self._frames)
        payload["num_episodes"] = int(self._episodes)
        if self.fmt == "zarr":
            root.attrs.update({key: _jsonable(value) for key, value in payload.items()})
        else:
            root.attrs["metadata_json"] = json.dumps(_jsonable(payload))
            root.close()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
