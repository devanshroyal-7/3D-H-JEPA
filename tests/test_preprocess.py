import numpy as np

from point_hjepa.data.preprocess import PreprocessConfig, farthest_point_sample, preprocess_pointclouds


def test_drops_invalid_points_transforms_and_keeps_objects():
    # World points. Base is at x=1, so base-frame x is world x - 1.
    xyzw = np.array(
        [
            [1.2, 0.0, 0.1, 1.0],  # object, inside workspace after the transform
            [0.0, 0.0, 0.1, 1.0],  # outside the workspace once moved into the base frame
            [1.2, 0.0, 0.2, 0.0],  # invalid homogeneous coordinate
            [1.15, 0.1, 0.0, 1.0],  # table point
            [1.3, -0.2, 0.05, 1.0],  # robot point
        ],
        dtype=np.float32,
    )
    seg = np.array([[5], [5], [5], [2], [1]], dtype=np.int16)
    rgb = np.arange(len(xyzw) * 3, dtype=np.uint8).reshape(len(xyzw), 3)
    out = preprocess_pointclouds(
        xyzw,
        seg,
        rgb=rgb,
        base_p=np.array([1.0, 0.0, 0.0]),
        base_q=np.array([1.0, 0.0, 0.0, 0.0]),
        robot_ids={1},
        object_ids={5},
        table_ids={2},
        config=PreprocessConfig(
            num_points=4,
            workspace_min=(-0.2, -0.7, -0.05),
            workspace_max=(1.0, 0.7, 0.8),
            table_keep=8,
            store_rgb=True,
            downsample="fps",
        ),
        device="cpu",
    )
    assert out["points"].shape == (1, 4, 3)
    assert out["rgb"].shape == (1, 4, 3)
    assert int(out["point_count"][0]) == 3
    # The invalid point and the out-of-workspace point are gone. Remaining
    # base-frame x values are world x - 1, all non-negative and small.
    real = out["points"][0, :3]
    assert np.all(real[:, 0] >= -1e-5)
    assert np.all(real[:, 0] < 0.5)


def test_batched_fps_shape_and_mask():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(2, 30, 3)).astype(np.float32)
    valid = np.ones((2, 30), dtype=bool)
    valid[1, 10:] = False
    import torch

    index = farthest_point_sample(torch.from_numpy(points), 8, torch.from_numpy(valid))
    assert tuple(index.shape) == (2, 8)
    assert index[1].max() < 10
