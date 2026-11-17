import numpy as np

from point_hjepa.data.dataset import PointCloudWindowDataset, valid_anchors
from point_hjepa.data.writer import AsyncEpisodeWriter


def _episode(fill: float, frames: int = 51, npoints: int = 8) -> dict:
    points = np.full((frames, npoints, 3), fill, dtype=np.float32)
    actions = np.full((frames, 7), fill, dtype=np.float32)
    actions[-1] = 0.0
    return {
        "points": points,
        "actions": actions,
        "goal_coords": np.full((frames, 3), fill, dtype=np.float32),
        "robot_state": np.full((frames, 4), fill, dtype=np.float32),
        "point_count": np.full(frames, npoints, dtype=np.int32),
        "success": np.zeros(frames, dtype=np.bool_),
        "done": np.array([False] * (frames - 1) + [True]),
        "source": 0 if fill == 0 else 1,
        "episode_success": bool(fill),
        "plan_failures": 0,
    }


def test_short_episodes_have_no_default_window():
    # Defaults need 20 frames of history at stride 10 and 20 frames of future.
    assert valid_anchors(30, context_frames=3, frameskip=5, horizon=4, level2_stride=10, level2_horizon=2) == []
    assert valid_anchors(51, context_frames=3, frameskip=5, horizon=4, level2_stride=10, level2_horizon=2)


def test_windows_do_not_cross_episode_boundaries(tmp_path):
    path = tmp_path / "episodes.zarr"
    writer = AsyncEpisodeWriter(path, "zarr")
    writer.write_episode(_episode(0.0))
    writer.write_episode(_episode(1.0))
    writer.close()
    dataset = PointCloudWindowDataset(path)
    assert len(dataset) > 0
    seen = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        episode = int(sample["episode_index"])
        seen.add(episode)
        blocks = [
            sample["context_points"],
            sample["target_points"],
            sample["level2_context_points"],
            sample["level2_target_points"],
        ]
        for block in blocks:
            assert set(np.unique(block.numpy()).tolist()) == {float(episode)}
        assert sample["context_points"].shape == (3, 8, 3)
        assert sample["context_actions"].shape == (2, 5, 7)
        assert sample["target_points"].shape == (4, 8, 3)
        assert sample["target_actions"].shape == (4, 5, 7)
        assert sample["level2_context_points"].shape == (3, 8, 3)
        assert sample["level2_context_actions"].shape == (2, 10, 7)
        assert sample["level2_target_points"].shape == (2, 8, 3)
        assert sample["level2_target_actions"].shape == (2, 10, 7)
        assert sample["goal_coords"].shape == (3,)
        start, end = dataset.episode_span(episode)
        assert end - start == 51
    assert seen == {0, 1}
    dataset.close()


def test_hdf5_roundtrip(tmp_path):
    path = tmp_path / "episodes.h5"
    writer = AsyncEpisodeWriter(path, "hdf5")
    writer.write_episode(_episode(0.25, frames=51))
    writer.close()
    dataset = PointCloudWindowDataset(
        path, context_frames=3, frameskip=5, horizon=4, level2_stride=10, level2_horizon=2
    )
    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["context_points"].shape[0] == 3
    dataset.close()
