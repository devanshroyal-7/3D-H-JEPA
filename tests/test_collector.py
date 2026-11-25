"""Unit and GPU integration tests for ManiSkill3 6D point-cloud data collection."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import pytest
import torch
import trimesh
import zarr

from point_hjepa.data.collect_maniskill3_6d import (
    DEFAULT_BOUNDS,
    SRMPExpertController,
    ZarrTrajectoryWriter,
    export_sample_plys,
    joint_target_to_delta_action,
    preprocess_pointcloud_batch,
    restore_snapshots_with_pd_sync,
)


class _DummyPose:
    def __init__(self, mat: torch.Tensor) -> None:
        self._mat = mat

    def inv(self) -> "_DummyPose":
        return _DummyPose(torch.linalg.inv(self._mat))

    def to_transformation_matrix(self) -> torch.Tensor:
        return self._mat


def test_preprocess_pointcloud_batch_subsample_and_pad() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Base pose at x = -0.615 -> x_base = x_world + 0.615
    mat = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    mat[:, 0, 3] = -0.615
    base_pose = _DummyPose(mat)

    N_raw = 8192
    xyzw = torch.zeros((2, N_raw, 4), device=device, dtype=torch.float32)
    rgb = torch.randint(0, 256, (2, N_raw, 3), device=device, dtype=torch.uint8)

    # Env 0: dense valid cloud well inside bounds (world x in [-0.3, 0.2] -> base x in [0.315, 0.815])
    xyzw[0, :, 0] = torch.linspace(-0.3, 0.2, N_raw, device=device)
    xyzw[0, :, 1] = torch.linspace(-0.3, 0.3, N_raw, device=device)
    xyzw[0, :, 2] = torch.linspace(0.0, 0.4, N_raw, device=device)
    xyzw[0, :, 3] = 1.0
    # Mark first 500 points invalid sky hits (w=0) and last 500 out of bounds
    xyzw[0, :500, 3] = 0.0
    xyzw[0, -500:, 0] = 5.0

    # Env 1: sparse cloud (only 120 valid points -> must pad with replacement to 4096)
    xyzw[1, :120, 0] = torch.linspace(-0.2, 0.1, 120, device=device)
    xyzw[1, :120, 1] = torch.linspace(-0.1, 0.1, 120, device=device)
    xyzw[1, :120, 2] = 0.1
    xyzw[1, :120, 3] = 1.0

    xyz_out, rgb_out = preprocess_pointcloud_batch(
        xyzw, rgb, base_pose, bounds=DEFAULT_BOUNDS, num_points=4096, voxel_size=0.0045
    )

    assert xyz_out.shape == (2, 4096, 3)
    assert rgb_out.shape == (2, 4096, 3)
    assert xyz_out.dtype == torch.float32
    assert rgb_out.dtype == torch.uint8

    for b in range(2):
        assert xyz_out[b, :, 0].min() >= DEFAULT_BOUNDS[0][0] - 1e-5
        assert xyz_out[b, :, 0].max() <= DEFAULT_BOUNDS[0][1] + 1e-5
        assert xyz_out[b, :, 1].min() >= DEFAULT_BOUNDS[1][0] - 1e-5
        assert xyz_out[b, :, 1].max() <= DEFAULT_BOUNDS[1][1] + 1e-5
        assert xyz_out[b, :, 2].min() >= DEFAULT_BOUNDS[2][0] - 1e-5
        assert xyz_out[b, :, 2].max() <= DEFAULT_BOUNDS[2][1] + 1e-5


def test_zarr_v3_writer_and_ply_export(tmp_path: Path) -> None:
    store_path = tmp_path / "sample.zarr"
    writer = ZarrTrajectoryWriter(store_path, num_points=64, action_dim=8, proprio_dim=26)

    for ep_len in (5, 7):
        writer.append_episode(
            xyz=np.random.randn(ep_len, 64, 3).astype(np.float32) * 0.1,
            rgb=np.random.randint(0, 256, (ep_len, 64, 3), dtype=np.uint8),
            action=np.random.uniform(-1, 1, (ep_len, 8)).astype(np.float32),
            proprio=np.random.randn(ep_len, 26).astype(np.float32),
            achieved_point=np.random.randn(ep_len, 3).astype(np.float32),
            ee_point=np.random.randn(ep_len, 3).astype(np.float32),
        )

    root = zarr.open_group(str(store_path), mode="r")
    assert root["xyz"].shape == (12, 64, 3)
    assert root["xyz"].dtype == np.float32
    assert root["rgb"].shape == (12, 64, 3)
    assert root["rgb"].dtype == np.uint8
    assert root["action"].shape == (12, 8)
    assert root["proprio"].shape == (12, 26)
    assert root["achieved_point"].shape == (12, 3)
    assert root["ee_point"].shape == (12, 3)
    assert np.array_equal(np.asarray(root["episode_ends"]), np.array([5, 12], dtype=np.int64))

    plys = export_sample_plys(store_path, tmp_path / "ply", count=2)
    assert len(plys) == 2
    for p in plys:
        loaded = trimesh.load(str(p))
        assert loaded.vertices.shape == (64, 3)


def test_joint_target_to_delta_action() -> None:
    q_curr = np.zeros(7, dtype=np.float32)
    q_targ_within = np.array([0.05, -0.1, 0.08, 0.0, 0.01, -0.05, 0.02], dtype=np.float32)
    act1 = joint_target_to_delta_action(q_curr, q_targ_within, gripper=-1.0, delta_scale=0.1)
    expected1 = np.array([0.5, -1.0, 0.8, 0.0, 0.1, -0.5, 0.2, -1.0], dtype=np.float32)
    assert act1.shape == (8,)
    assert np.allclose(act1, expected1, atol=1e-6)

    q_targ_scaled = np.array([0.05, -0.1, 0.2, 0.0, 0.01, -0.05, 0.1], dtype=np.float32)
    act2 = joint_target_to_delta_action(q_curr, q_targ_scaled, gripper=-1.0, delta_scale=0.1)
    expected2 = np.array([0.25, -0.5, 1.0, 0.0, 0.05, -0.25, 0.5, -1.0], dtype=np.float32)
    assert act2.shape == (8,)
    assert np.allclose(act2, expected2, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for physx_cuda")
def test_gpu_snapshot_restore_pd_sync_and_srmp_plan() -> None:
    env = gym.make(
        "StackCube-v1",
        num_envs=2,
        obs_mode="pointcloud",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cuda",
        render_mode=None,
        max_episode_steps=100,
    )
    env.reset(seed=7)
    u = env.unwrapped

    # Step with random delta to move arm away from initial pose
    for _ in range(5):
        env.step(torch.full((2, 8), 0.5, device=u.device))

    snap = u.get_state_dict()
    saved_qpos = u.agent.robot.get_qpos().clone()

    # Perturb further
    for _ in range(5):
        env.step(torch.full((2, 8), -0.8, device=u.device))

    # Restore and verify arm stays at saved_qpos after a zero-delta step
    restore_snapshots_with_pd_sync(env, snap)
    restored_qpos = u.agent.robot.get_qpos()
    assert torch.allclose(restored_qpos, saved_qpos, atol=1e-4)

    zero_act = torch.zeros((2, 8), device=u.device)
    zero_act[:, 7] = 1.0
    env.step(zero_act)
    after_step_qpos = u.agent.robot.get_qpos()[:, :7]
    assert torch.max(torch.abs(after_step_qpos - saved_qpos[:, :7])).item() < 0.05
    env.close()

    # Verify SRMP plans end-to-end on StackCube-v1 and PullCubeTool-v1
    for tid in ("StackCube-v1", "PullCubeTool-v1"):
        e1 = gym.make(
            tid,
            num_envs=1,
            obs_mode="pointcloud",
            control_mode="pd_joint_delta_pos",
            sim_backend="physx_cuda",
            render_mode=None,
            max_episode_steps=100,
        )
        stages = None
        for seed in range(5):
            e1.reset(seed=seed)
            expert = SRMPExpertController(e1, tid)
            stages = expert.plan_episode()
            if stages is not None:
                break
        e1.close()
        assert stages is not None and len(stages) >= 4
