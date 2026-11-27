"""GPU-parallel ManiSkill3 6D RGB-XYZ point-cloud collector for Point-H-JEPA.

Supports two dataset modes using the 8D ``pd_joint_delta_pos`` action space
(7 normalized Panda arm joint deltas in [-0.1, 0.1] rad/step + 1 gripper):

1. ``expert_srmp``: Clean collision-free SRMP (VAMP) demonstrations on
   ``StackCube-v1`` and ``PullCubeTool-v1``, saving ``get_state_dict()``
   snapshots at key trajectory stages (approach, grasp, lift, transport,
   place_or_pull).
2. ``branched_exploratory``: GPU-parallel ``physx_cuda`` rollouts restoring
   SRMP snapshots via ``set_state_dict(...)`` (with immediate PD drive target
   synchronization) and branching into a configurable exploration mixture:
   - 50% noisy expert (Gaussian action noise sigma in [0.05, 0.15])
   - 30% interrupted (expert -> uniform random at t_split, mid-grasp
     gripper-held OU motion, release/drop motion)
   - 20% temporally correlated Ornstein-Uhlenbeck (OU) exploration

All point clouds are unsegmented (tabletop, robot arm, targets, distractors
retained; segmentation masks are never used to filter scene geometry) and stored
in robot-base metric coordinates (float32) + [0, 255] RGB (uint8) in Zarr v3.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import hydra
import mani_skill.envs  # noqa: F401
import numpy as np
import srmp
import torch
import trimesh
import zarr
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from point_hjepa.data.zarr_writer import ZarrTrajectoryWriter

log = logging.getLogger(__name__)

ARM_DELTA_SCALE = 0.1
DEFAULT_BOUNDS = ((-0.15, 1.05), (-0.55, 0.55), (-0.05, 0.65))


# --------------------------------------------------------------------------- #
# Geometry & on-GPU point-cloud preprocessing
# --------------------------------------------------------------------------- #


def _world_to_base_matrix(base_pose: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    """Return ``T_base^{-1}`` with shape ``[B, 4, 4]`` on ``device``."""
    inv_mat = base_pose.inv().to_transformation_matrix()
    if not isinstance(inv_mat, torch.Tensor):
        inv_mat = torch.as_tensor(inv_mat, dtype=torch.float32, device=device)
    inv_mat = inv_mat.to(device=device, dtype=torch.float32)
    if inv_mat.dim() == 2:
        inv_mat = inv_mat.unsqueeze(0)
    if inv_mat.shape[0] == 1 and batch_size > 1:
        inv_mat = inv_mat.expand(batch_size, -1, -1)
    return inv_mat


def _transform_points(pts: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    """Apply rigid ``[4, 4]`` transform ``mat`` to ``[N, 3]`` points."""
    return pts @ mat[:3, :3].T + mat[:3, 3]


def _transform_points_batch(pts: torch.Tensor, mats: torch.Tensor) -> torch.Tensor:
    """Apply batched ``[B, 4, 4]`` transforms to ``[B, 3]`` points."""
    return torch.bmm(mats[:, :3, :3], pts.unsqueeze(-1)).squeeze(-1) + mats[:, :3, 3]


def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product for wxyz quaternions ``[B, 4]``."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _pose_to_mat(pose: Any) -> np.ndarray:
    """Convert an srmp.Pose or sapien.Pose (p, q_wxyz) into a 4x4 numpy matrix."""
    p = np.asarray(pose.p, dtype=np.float64).reshape(3)
    w, x, y, z = np.asarray(pose.q, dtype=np.float64).reshape(4)
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _mat_to_srmp_pose(T: np.ndarray) -> srmp.Pose:
    return srmp.Pose(T[:3, 3].copy(), _rotmat_to_quat_wxyz(T[:3, :3]))


@torch.no_grad()
def preprocess_pointcloud_batch(
    xyzw: torch.Tensor,
    rgb: torch.Tensor,
    base_pose: Any,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = DEFAULT_BOUNDS,
    num_points: int = 4096,
    voxel_size: float = 0.0045,
    generator: torch.Generator | None = None,
    sort_by_voxel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preprocess raw ManiSkill3 point clouds on GPU without segmentation filtering.

    Steps per environment ``b``:
    1. Filter valid rays ``xyzw[b, :, 3] > 0.5``.
    2. Transform world coordinates into robot-base frame via ``T_base^{-1}``.
    3. Crop to ``bounds`` in robot-base meters.
    4. Voxel-grid deduplicate (``voxel_size``) and uniform-subsample or pad with
       replacement to exact ``num_points``.
    5. Return raw metric ``xyz`` (``float32``) and ``rgb`` in ``[0, 255]`` (``uint8``).

    With ``sort_by_voxel`` the sampled points are kept in voxel-key order (a
    lossless reordering of the set) so the Zarr compressor sees locally smooth
    coordinates.
    """
    device = xyzw.device
    B = xyzw.shape[0]
    t_inv = _world_to_base_matrix(base_pose, B, device)

    b_min = torch.tensor([bounds[0][0], bounds[1][0], bounds[2][0]], device=device, dtype=torch.float32)
    b_max = torch.tensor([bounds[0][1], bounds[1][1], bounds[2][1]], device=device, dtype=torch.float32)
    dims = torch.ceil((b_max - b_min) / max(voxel_size, 1e-4)).to(torch.int64) + 1
    stride_y = int(dims[2].item())
    stride_x = int((dims[1] * dims[2]).item())

    out_xyz = torch.zeros((B, num_points, 3), device=device, dtype=torch.float32)
    out_rgb = torch.zeros((B, num_points, 3), device=device, dtype=torch.uint8)

    for b in range(B):
        valid = xyzw[b, :, 3] > 0.5
        if not torch.any(valid):
            continue
        pts = _transform_points(xyzw[b, valid, :3].float(), t_inv[b])
        cols = rgb[b, valid, :3].to(torch.uint8)

        in_box = torch.all((pts >= b_min) & (pts <= b_max), dim=-1)
        if not torch.any(in_box):
            continue
        pts = pts[in_box]
        cols = cols[in_box]

        if voxel_size > 0.0 and pts.shape[0] > 1:
            vox = torch.floor((pts - b_min) / voxel_size).to(torch.int64)
            keys = vox[:, 0] * stride_x + vox[:, 1] * stride_y + vox[:, 2]
            order = torch.argsort(keys)
            sorted_keys = keys[order]
            keep = torch.ones_like(sorted_keys, dtype=torch.bool)
            keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
            idx = order[keep]
            pts = pts[idx]
            cols = cols[idx]

        n_avail = pts.shape[0]
        if n_avail >= num_points:
            sel = torch.randperm(n_avail, device=device, generator=generator)[:num_points]
        else:
            sel = torch.randint(0, n_avail, (num_points,), device=device, generator=generator)
        if sort_by_voxel:
            sel = torch.sort(sel).values

        out_xyz[b] = pts[sel]
        out_rgb[b] = cols[sel]

    return out_xyz, out_rgb


@torch.no_grad()
def extract_step_metadata(env: Any, task_id: str) -> dict[str, torch.Tensor]:
    """Extract robot-base frame proprioception, ``ee_point``, and ``achieved_point``."""
    u = env.unwrapped
    device = u.device
    qpos = u.agent.robot.get_qpos().float()
    qvel = u.agent.robot.get_qvel().float()
    B = qpos.shape[0]

    t_inv = _world_to_base_matrix(u.agent.robot.pose, B, device)
    base_q_inv = u.agent.robot.pose.inv().q.to(device=device, dtype=torch.float32)
    if base_q_inv.dim() == 1:
        base_q_inv = base_q_inv.unsqueeze(0)
    if base_q_inv.shape[0] == 1 and B > 1:
        base_q_inv = base_q_inv.expand(B, -1)

    tcp_p_base = _transform_points_batch(u.agent.tcp.pose.p.float(), t_inv)
    tcp_q_base = _quat_mul_wxyz(base_q_inv, u.agent.tcp.pose.q.float())
    gripper_width = qpos[:, -2:].sum(dim=-1, keepdim=True)
    proprio = torch.cat([qpos, qvel, tcp_p_base, tcp_q_base, gripper_width], dim=-1)

    if task_id == "StackCube-v1":
        target_world = u.cubeA.pose.p.float()
    elif task_id == "PullCubeTool-v1":
        target_world = u.cube.pose.p.float()
    else:
        raise ValueError(f"Unsupported task_id: {task_id!r}")

    achieved_point = _transform_points_batch(target_world, t_inv)
    return {
        "proprio": proprio,
        "ee_point": tcp_p_base,
        "achieved_point": achieved_point,
    }


# --------------------------------------------------------------------------- #
# Action conversion & PD state-dict synchronization
# --------------------------------------------------------------------------- #


def joint_target_to_delta_action(
    q_curr: np.ndarray | torch.Tensor,
    q_target: np.ndarray | torch.Tensor,
    gripper: float | torch.Tensor,
    delta_scale: float = ARM_DELTA_SCALE,
) -> np.ndarray | torch.Tensor:
    """Convert target arm joint configuration into normalized 8D ``pd_joint_delta_pos`` action."""
    if isinstance(q_curr, torch.Tensor):
        q_t = torch.as_tensor(q_target, device=q_curr.device, dtype=torch.float32)
        diff = q_t - q_curr.float()
        scale = torch.clamp(diff.abs().amax(dim=-1, keepdim=True), min=delta_scale)
        arm = diff / scale
        grip = torch.as_tensor(gripper, device=q_curr.device, dtype=torch.float32).reshape(-1, 1)
        if arm.dim() == 1:
            return torch.cat([arm, grip.reshape(1)], dim=-1)
        return torch.cat([arm, grip.expand(arm.shape[0], 1)], dim=-1)

    q_t = np.asarray(q_target, dtype=np.float32)
    diff = q_t - np.asarray(q_curr, dtype=np.float32)
    scale = np.maximum(delta_scale, np.max(np.abs(diff), axis=-1, keepdims=True))
    arm = diff / scale
    if arm.ndim == 1:
        return np.concatenate([arm, np.array([float(gripper)], dtype=np.float32)])
    grip = np.full((arm.shape[0], 1), float(gripper), dtype=np.float32)
    return np.concatenate([arm, grip], axis=-1)


def restore_snapshots_with_pd_sync(
    env: Any,
    batched_state_dict: dict[str, Any],
    env_idx: torch.Tensor | None = None,
) -> None:
    """Restore ``env.unwrapped.set_state_dict`` and synchronize PD controller targets."""
    u = env.unwrapped
    u.set_state_dict(batched_state_dict, env_idx=env_idx)

    qpos = u.agent.robot.get_qpos()
    ctrls = u.agent.controller.controllers
    for name, ctrl in ctrls.items():
        idx = ctrl.active_joint_indices
        q_slice = qpos[:, idx].clone() if env_idx is None else qpos[env_idx][:, idx].clone()
        if env_idx is None:
            ctrl._start_qpos = q_slice.clone()
            ctrl._target_qpos = q_slice.clone()
            ctrl.set_drive_targets(q_slice)
        else:
            if ctrl._start_qpos is None:
                ctrl._start_qpos = qpos[:, idx].clone()
            else:
                ctrl._start_qpos[env_idx] = q_slice
            if ctrl._target_qpos is None:
                ctrl._target_qpos = qpos[:, idx].clone()
            else:
                ctrl._target_qpos[env_idx] = q_slice
            ctrl.set_drive_targets(ctrl._target_qpos)

    if u.gpu_sim_enabled:
        u.scene._gpu_apply_all()
        u.scene.px.gpu_update_articulation_kinematics()
        u.scene._gpu_fetch_all()


# --------------------------------------------------------------------------- #
# SRMP (VAMP) expert planner for StackCube-v1 and PullCubeTool-v1
# --------------------------------------------------------------------------- #


@dataclass
class PlannedStage:
    name: str
    waypoints: np.ndarray  # [K, 7]
    gripper: float  # +1.0 open, -1.0 close
    settle_steps: int = 0
    is_grasped: bool = False


class SRMPExpertController:
    """Plans collision-free Panda joint trajectories using SRMP (VAMP) on ManiSkill3."""

    def __init__(self, env: Any, task_id: str) -> None:
        self.env = env
        self.task_id = task_id
        u = env.unwrapped
        self.hand_link = [l for l in u.agent.robot.get_links() if l.name == "panda_hand"][0]

        T_hand = _pose_to_mat(
            srmp.Pose(
                self.hand_link.pose.p[0].cpu().numpy(),
                self.hand_link.pose.q[0].cpu().numpy(),
            )
        )
        T_tcp = _pose_to_mat(
            srmp.Pose(
                u.agent.tcp.pose.p[0].cpu().numpy(),
                u.agent.tcp.pose.q[0].cpu().numpy(),
            )
        )
        self.T_tcp_to_hand = np.linalg.inv(T_tcp) @ T_hand

        self.planner = srmp.PlannerInterface()
        self.robot_name = self.planner.add_robot("panda")
        base_p = u.agent.robot.pose.p[0].cpu().numpy()
        base_q = u.agent.robot.pose.q[0].cpu().numpy()
        self.planner.set_base_pose(self.robot_name, srmp.Pose(base_p, base_q))
        self.planner.set_base_contact_allowed(self.robot_name, True)

        # Table top sits at world z = 0; cut out under Panda base (x < -0.50) and
        # offset top down 5 mm so VAMP base-link spheres never clip the table.
        self.planner.add_box(
            "table",
            np.array([1.05, 2.4, 0.2], dtype=np.float64),
            srmp.Pose(np.array([0.025, 0.0, -0.105]), np.array([1.0, 0.0, 0.0, 0.0])),
        )
        self.dynamic_objects: list[str] = []

    def _clear_dynamic_objects(self) -> None:
        self.planner.detach_all_objects(also_remove=False)
        for name in self.dynamic_objects:
            if name in self.planner.get_object_names():
                self.planner.remove_object(name)
        self.dynamic_objects.clear()

    def _tcp_to_hand_pose(self, tcp_p: np.ndarray, tcp_q_wxyz: np.ndarray) -> srmp.Pose:
        T_tcp = _pose_to_mat(srmp.Pose(tcp_p, tcp_q_wxyz))
        return _mat_to_srmp_pose(T_tcp @ self.T_tcp_to_hand)

    @staticmethod
    def _densify_waypoints(waypoints: np.ndarray, max_joint_step: float = 0.075) -> np.ndarray:
        """Resample joint waypoints so each consecutive step fits within ``0.1`` rad/step."""
        if len(waypoints) <= 1:
            return waypoints.astype(np.float32)
        out = [waypoints[0]]
        for nxt in waypoints[1:]:
            prev = out[-1]
            diff = nxt - prev
            n_sub = max(1, int(np.ceil(np.max(np.abs(diff)) / max_joint_step)))
            for s in range(1, n_sub + 1):
                out.append(prev + (s / n_sub) * diff)
        return np.asarray(out, dtype=np.float32)

    def _plan_motion(self, q_start: np.ndarray, target_hand_pose: srmp.Pose) -> np.ndarray | None:
        self.planner.set_qpos(self.robot_name, q_start)
        q_goal, status = self.planner.collision_aware_ik(
            self.robot_name, target_hand_pose, q_start, timeout=0.15, best_of=True
        )
        if status != "found" or q_goal is None:
            return None

        self.planner.set_qpos(self.robot_name, q_start)
        traj = None
        try:
            self.planner.make_planner(
                [self.robot_name],
                {
                    "planner_id": "vamp_rrtc",
                    "validity_backend": "auto",
                    "time_limit": "2.0",
                    "num_threads": "4",
                },
            )
            if self.planner.get_active_validity_backend() == "vamp":
                traj = self.planner.plan(
                    q_start.tolist(),
                    srmp.GoalConstraint(srmp.GoalType.JOINTS, [list(q_goal)]),
                )
        except RuntimeError:
            traj = None

        if traj is None or len(traj.positions) == 0:
            self.planner.set_qpos(self.robot_name, q_start)
            self.planner.make_planner(
                [self.robot_name],
                {
                    "planner_id": "wAstar",
                    "validity_backend": "coal",
                    "weight": "50",
                    "time_limit": "3.0",
                },
            )
            traj = self.planner.plan(
                q_start.tolist(),
                srmp.GoalConstraint(srmp.GoalType.JOINTS, [list(q_goal)]),
            )
        if len(traj.positions) == 0:
            return None
        return self._densify_waypoints(np.asarray(traj.positions, dtype=np.float64))

    def _plan_screw_segment(
        self, q_start: np.ndarray, target_hand_pose: srmp.Pose, collision_aware: bool = False
    ) -> np.ndarray | None:
        try:
            traj = self.planner.plan_screw(
                self.robot_name,
                "panda_hand",
                start_qpos=q_start,
                end_pose=target_hand_pose,
                qpos_step=0.075,
                max_steps=400,
                collision_aware=collision_aware,
            )
            if len(traj.positions) == 0:
                return None
            return self._densify_waypoints(np.asarray(traj.positions, dtype=np.float64))
        except RuntimeError:
            return None

    def _aligned_topdown_quat(self, obj_q_wxyz: np.ndarray, tcp_q_wxyz: np.ndarray) -> np.ndarray:
        """Return top-down (-Z approach) TCP quaternion aligned with object's XY axes."""
        T_obj = _pose_to_mat(srmp.Pose(np.zeros(3), obj_q_wxyz))
        T_tcp = _pose_to_mat(srmp.Pose(np.zeros(3), tcp_q_wxyz))
        tcp_y = T_tcp[:3, 1]
        candidates = [T_obj[:3, 0], -T_obj[:3, 0], T_obj[:3, 1], -T_obj[:3, 1]]
        closing = max(candidates, key=lambda c: float(np.dot(c[:2], tcp_y[:2])))
        closing = np.array([closing[0], closing[1], 0.0], dtype=np.float64)
        closing /= np.linalg.norm(closing) + 1e-8
        z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        x_axis = np.cross(closing, z_axis)
        x_axis /= np.linalg.norm(x_axis) + 1e-8
        y_axis = np.cross(z_axis, x_axis)
        R = np.column_stack([x_axis, y_axis, z_axis])
        return _rotmat_to_quat_wxyz(R)

    def plan_episode(self) -> list[PlannedStage] | None:
        self._clear_dynamic_objects()
        u = self.env.unwrapped
        q_curr = u.agent.robot.get_qpos()[0, :7].cpu().numpy().astype(np.float64)
        tcp_q_now = u.agent.tcp.pose.q[0].cpu().numpy().astype(np.float64)
        self.planner.set_gripper_qpos(self.robot_name, np.array([0.04, 0.04]))

        if self.task_id == "StackCube-v1":
            return self._plan_stack_cube(q_curr, tcp_q_now)
        if self.task_id == "PullCubeTool-v1":
            return self._plan_pull_cube_tool(q_curr, tcp_q_now)
        raise ValueError(f"Unsupported task_id: {self.task_id!r}")

    def _plan_stack_cube(self, q_curr: np.ndarray, tcp_q_now: np.ndarray) -> list[PlannedStage] | None:
        u = self.env.unwrapped
        pA = u.cubeA.pose.p[0].cpu().numpy().astype(np.float64)
        qA = u.cubeA.pose.q[0].cpu().numpy().astype(np.float64)
        pB = u.cubeB.pose.p[0].cpu().numpy().astype(np.float64)
        qB = u.cubeB.pose.q[0].cpu().numpy().astype(np.float64)

        self.planner.add_box("cubeA", np.array([0.04, 0.04, 0.04]), srmp.Pose(pA, qA))
        self.planner.add_box("cubeB", np.array([0.04, 0.04, 0.04]), srmp.Pose(pB, qB))
        self.dynamic_objects.extend(["cubeA", "cubeB"])

        grasp_q = self._aligned_topdown_quat(qA, tcp_q_now)
        pre_grasp = self._tcp_to_hand_pose(pA + np.array([0.0, 0.0, 0.075]), grasp_q)
        grasp_pose = self._tcp_to_hand_pose(pA + np.array([0.0, 0.0, 0.002]), grasp_q)
        lift_pose = self._tcp_to_hand_pose(pA + np.array([0.0, 0.0, 0.11]), grasp_q)
        pre_place = self._tcp_to_hand_pose(pB + np.array([0.0, 0.0, 0.105]), grasp_q)
        place_pose = self._tcp_to_hand_pose(pB + np.array([0.0, 0.0, 0.043]), grasp_q)

        stages: list[PlannedStage] = []

        wp_app = self._plan_motion(q_curr, pre_grasp)
        if wp_app is None:
            return None
        stages.append(PlannedStage("approach", wp_app, gripper=1.0, settle_steps=1, is_grasped=False))
        q_curr = wp_app[-1].astype(np.float64)

        wp_grasp = self._plan_screw_segment(q_curr, grasp_pose, collision_aware=False)
        if wp_grasp is None:
            return None
        stages.append(PlannedStage("grasp", wp_grasp, gripper=1.0, settle_steps=0, is_grasped=False))
        q_curr = wp_grasp[-1].astype(np.float64)
        stages.append(
            PlannedStage("close_grip", q_curr[None, :].astype(np.float32), gripper=-1.0, settle_steps=6, is_grasped=True)
        )

        self.planner.set_qpos(self.robot_name, q_curr)
        self.planner.attach_object("cubeA", self.robot_name, "panda_hand", is_link_name=True)

        wp_lift = self._plan_screw_segment(q_curr, lift_pose, collision_aware=False)
        if wp_lift is None:
            return None
        stages.append(PlannedStage("lift", wp_lift, gripper=-1.0, settle_steps=1, is_grasped=True))
        q_curr = wp_lift[-1].astype(np.float64)

        wp_trans = self._plan_motion(q_curr, pre_place)
        if wp_trans is None:
            return None
        stages.append(PlannedStage("transport", wp_trans, gripper=-1.0, settle_steps=1, is_grasped=True))
        q_curr = wp_trans[-1].astype(np.float64)

        wp_place = self._plan_screw_segment(q_curr, place_pose, collision_aware=False)
        if wp_place is None:
            return None
        stages.append(PlannedStage("place_or_pull", wp_place, gripper=-1.0, settle_steps=1, is_grasped=True))
        q_curr = wp_place[-1].astype(np.float64)
        stages.append(
            PlannedStage("open_grip", q_curr[None, :].astype(np.float32), gripper=1.0, settle_steps=6, is_grasped=False)
        )
        return stages

    def _plan_pull_cube_tool(self, q_curr: np.ndarray, tcp_q_now: np.ndarray) -> list[PlannedStage] | None:
        u = self.env.unwrapped
        p_tool = u.l_shape_tool.pose.p[0].cpu().numpy().astype(np.float64)
        q_tool = u.l_shape_tool.pose.q[0].cpu().numpy().astype(np.float64)
        p_cube = u.cube.pose.p[0].cpu().numpy().astype(np.float64)
        q_cube = u.cube.pose.q[0].cpu().numpy().astype(np.float64)

        # Register L-shaped tool as a single merged mesh so VAMP supports attaching it.
        m_handle = trimesh.creation.box(extents=[0.20, 0.05, 0.05])
        m_handle.apply_translation([0.10, 0.0, 0.0])
        m_hook = trimesh.creation.box(extents=[0.05, 0.10, 0.05])
        m_hook.apply_translation([0.175, 0.05, 0.0])
        m_tool = trimesh.util.concatenate([m_handle, m_hook])
        self.planner.add_mesh(
            "l_shape_tool",
            vertices=np.asarray(m_tool.vertices, dtype=np.float64),
            triangles=np.asarray(m_tool.faces, dtype=np.int32),
            pose=srmp.Pose(p_tool, q_tool),
        )
        self.planner.add_box("cube", np.array([0.04, 0.04, 0.04]), srmp.Pose(p_cube, q_cube))
        self.dynamic_objects.extend(["l_shape_tool", "cube"])

        grasp_q = self._aligned_topdown_quat(q_tool, tcp_q_now)
        grasp_p = p_tool + np.array([0.03, 0.0, 0.0], dtype=np.float64)

        pre_grasp = self._tcp_to_hand_pose(grasp_p + np.array([0.0, 0.0, 0.07]), grasp_q)
        grasp_pose = self._tcp_to_hand_pose(grasp_p, grasp_q)
        lift_pose = self._tcp_to_hand_pose(grasp_p + np.array([0.0, 0.0, 0.18]), grasp_q)

        behind_p = np.array([p_cube[0] - 0.070, p_cube[1] - 0.055, 0.028], dtype=np.float64)
        approach_behind = self._tcp_to_hand_pose(behind_p + np.array([0.0, 0.0, 0.14]), grasp_q)
        lower_hook = self._tcp_to_hand_pose(behind_p, grasp_q)

        stages: list[PlannedStage] = []

        wp_app = self._plan_motion(q_curr, pre_grasp)
        if wp_app is None:
            return None
        stages.append(PlannedStage("approach", wp_app, gripper=1.0, settle_steps=1, is_grasped=False))
        q_curr = wp_app[-1].astype(np.float64)

        wp_grasp = self._plan_screw_segment(q_curr, grasp_pose, collision_aware=False)
        if wp_grasp is None:
            return None
        stages.append(PlannedStage("grasp", wp_grasp, gripper=1.0, settle_steps=3, is_grasped=False))
        q_curr = wp_grasp[-1].astype(np.float64)
        stages.append(
            PlannedStage("close_grip", q_curr[None, :].astype(np.float32), gripper=-1.0, settle_steps=7, is_grasped=True)
        )

        self.planner.set_qpos(self.robot_name, q_curr)
        self.planner.attach_object("l_shape_tool", self.robot_name, "panda_hand", is_link_name=True)

        wp_lift = self._plan_screw_segment(q_curr, lift_pose, collision_aware=False)
        if wp_lift is None:
            return None
        stages.append(PlannedStage("lift", wp_lift, gripper=-1.0, settle_steps=1, is_grasped=True))
        q_curr = wp_lift[-1].astype(np.float64)

        wp_trans = self._plan_motion(q_curr, approach_behind)
        if wp_trans is None:
            return None
        stages.append(PlannedStage("transport", wp_trans, gripper=-1.0, settle_steps=1, is_grasped=True))
        q_curr = wp_trans[-1].astype(np.float64)

        wp_lower = self._plan_screw_segment(q_curr, lower_hook, collision_aware=False)
        if wp_lower is None:
            return None
        stages.append(PlannedStage("lower_hook", wp_lower, gripper=-1.0, settle_steps=4, is_grasped=True))
        q_curr = wp_lower[-1].astype(np.float64)

        wp_pull = None
        for pull_dx in (0.42, 0.38, 0.34, 0.30):
            pull_pose = self._tcp_to_hand_pose(behind_p + np.array([-pull_dx, 0.0, 0.0]), grasp_q)
            wp_pull = self._plan_screw_segment(q_curr, pull_pose, collision_aware=False)
            if wp_pull is not None:
                wp_pull = self._densify_waypoints(wp_pull, max_joint_step=0.05)
                break
        if wp_pull is None:
            return None
        stages.append(PlannedStage("place_or_pull", wp_pull, gripper=-1.0, settle_steps=4, is_grasped=True))
        return stages


# --------------------------------------------------------------------------- #
# Dataset 1: Clean expert SRMP collector + stage snapshot export
# --------------------------------------------------------------------------- #


def _squeeze_state_dict_to_cpu(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        group: {name: tensor[0].detach().cpu().clone() for name, tensor in items.items()}
        for group, items in state_dict.items()
    }


def _stack_state_dicts(snapshots: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in snapshots[0]:
        out[group] = {}
        for name in snapshots[0][group]:
            out[group][name] = torch.stack([s[group][name] for s in snapshots], dim=0).to(device)
    return out


def _flatten_stages(stages: list[PlannedStage]) -> tuple[np.ndarray, np.ndarray]:
    """Expand stages into dense ``(waypoints [M, 7], grippers [M])`` arrays."""
    wps, grips = [], []
    for st in stages:
        for wp in st.waypoints:
            wps.append(wp)
            grips.append(st.gripper)
        for _ in range(st.settle_steps):
            wps.append(st.waypoints[-1])
            grips.append(st.gripper)
    return np.asarray(wps, dtype=np.float32), np.asarray(grips, dtype=np.float32)


def _disk_space_ok(path: Path, min_free_gb: float) -> bool:
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < min_free_gb:
        log.warning("Only %.1f GB free at %s (< min_free_gb=%.1f); stopping collection early.", free_gb, path, min_free_gb)
        return False
    return True


def _bounds_from_cfg(cfg: DictConfig) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    wb = cfg.workspace_bounds
    return (
        (float(wb.x[0]), float(wb.x[1])),
        (float(wb.y[0]), float(wb.y[1])),
        (float(wb.z[0]), float(wb.z[1])),
    )


def collect_expert_srmp_dataset(cfg: DictConfig) -> tuple[Path, Path]:
    """Collect clean SRMP demonstrations and export stage snapshots for branching."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = out_dir / f"{cfg.task_id}_expert_srmp.zarr"
    snap_path = out_dir / f"{cfg.task_id}_expert_srmp_snapshots.pt"
    bounds = _bounds_from_cfg(cfg)

    env = gym.make(
        cfg.task_id,
        num_envs=1,
        obs_mode="pointcloud",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cuda",
        render_mode=None,
        max_episode_steps=int(cfg.max_episode_steps),
    )
    expert = SRMPExpertController(env, cfg.task_id)
    writer = ZarrTrajectoryWriter(
        zarr_path,
        num_points=int(cfg.num_points),
        action_dim=int(cfg.action_dim),
        attrs={
            "task_id": str(cfg.task_id),
            "control_mode": str(cfg.control_mode),
            "action_dim": int(cfg.action_dim),
            "dataset_type": "expert_srmp",
            "workspace_bounds": [list(b) for b in bounds],
            "xyz_dtype": str(cfg.get("xyz_dtype", "float32")),
        },
        xyz_dtype=str(cfg.get("xyz_dtype", "float32")),
    )

    device = env.unwrapped.device
    gen = torch.Generator(device=device).manual_seed(int(cfg.seed))
    saved_snapshots: list[dict[str, Any]] = []
    target_eps = int(cfg.expert_episodes)
    collected = 0
    success_count = 0
    attempt = 0

    key_stage_names = {"approach", "grasp", "lift", "transport", "place_or_pull"}
    pbar = trange(target_eps, desc=f"expert_srmp ({cfg.task_id})")
    min_free_gb = float(cfg.get("min_free_gb", 0.0))
    while collected < target_eps:
        if collected % 25 == 0 and not _disk_space_ok(out_dir, min_free_gb):
            break
        obs, _ = env.reset(seed=int(cfg.seed) + attempt)
        attempt += 1
        stages = expert.plan_episode()
        if stages is None:
            continue

        ep_xyz, ep_rgb, ep_act, ep_prop, ep_ach, ep_ee = [], [], [], [], [], []
        ep_snapshots: list[dict[str, Any]] = []
        aborted = False
        step_idx = 0

        for s_idx, stage in enumerate(stages):
            # Capture snapshot right at the beginning of key stages.
            if stage.name in key_stage_names:
                rem_wps, rem_grips = _flatten_stages(stages[s_idx:])
                ep_snapshots.append(
                    {
                        "stage": stage.name,
                        "is_grasped": bool(stage.is_grasped),
                        "state_dict": _squeeze_state_dict_to_cpu(env.unwrapped.get_state_dict()),
                        "remaining_waypoints": rem_wps,
                        "remaining_grippers": rem_grips,
                    }
                )

            targets = list(stage.waypoints) + [stage.waypoints[-1]] * stage.settle_steps

            def _step_to_target(wp_target: np.ndarray) -> bool:
                nonlocal obs, step_idx
                if step_idx >= int(cfg.max_episode_steps):
                    return True
                q_curr = env.unwrapped.agent.robot.get_qpos()[0, :7].cpu().numpy()
                act_np = joint_target_to_delta_action(q_curr, wp_target, stage.gripper, float(cfg.arm_delta_scale))
                act_t = torch.from_numpy(act_np).unsqueeze(0).to(device)

                xyz_b, rgb_b = preprocess_pointcloud_batch(
                    obs["pointcloud"]["xyzw"],
                    obs["pointcloud"]["rgb"],
                    env.unwrapped.agent.robot.pose,
                    bounds=bounds,
                    num_points=int(cfg.num_points),
                    voxel_size=float(cfg.voxel_size),
                    generator=gen,
                    sort_by_voxel=bool(cfg.get("sort_by_voxel", False)),
                )
                meta = extract_step_metadata(env, cfg.task_id)

                ep_xyz.append(xyz_b[0].cpu().numpy())
                ep_rgb.append(rgb_b[0].cpu().numpy())
                ep_act.append(act_np)
                ep_prop.append(meta["proprio"][0].cpu().numpy())
                ep_ach.append(meta["achieved_point"][0].cpu().numpy())
                ep_ee.append(meta["ee_point"][0].cpu().numpy())

                obs, _, terminated, truncated, _ = env.step(act_t)
                step_idx += 1
                return bool(terminated[0] or truncated[0])

            done = False
            max_wp_tries = 3 if stage.name == "place_or_pull" else 1
            for wp in targets:
                for _ in range(max_wp_tries):
                    done = _step_to_target(wp)
                    if done:
                        break
                    q_curr = env.unwrapped.agent.robot.get_qpos()[0, :7].cpu().numpy()
                    if float(np.max(np.abs(q_curr - wp))) <= 0.035:
                        break
                if done:
                    break

            if not done and stage.name in {"grasp", "lower_hook", "place_or_pull"}:
                final_wp = stage.waypoints[-1]
                for _ in range(8):
                    q_curr = env.unwrapped.agent.robot.get_qpos()[0, :7].cpu().numpy()
                    if float(np.max(np.abs(q_curr - final_wp))) < 0.015:
                        break
                    done = _step_to_target(final_wp)
                    if done:
                        break

            if done or step_idx >= int(cfg.max_episode_steps):
                break

        if not aborted and len(ep_xyz) > 0:
            writer.append_episode(
                np.stack(ep_xyz),
                np.stack(ep_rgb),
                np.stack(ep_act),
                np.stack(ep_prop),
                np.stack(ep_ach),
                np.stack(ep_ee),
            )
            saved_snapshots.extend(ep_snapshots)
            eval_info = env.unwrapped.evaluate()
            if bool(eval_info.get("success", torch.tensor([False]))[0]):
                success_count += 1
            collected += 1
            pbar.update(1)

        if attempt > max(target_eps * 5, 50) and collected == 0:
            raise RuntimeError(f"SRMP planner failed on {attempt} consecutive attempts for {cfg.task_id}")

    pbar.close()
    env.close()
    torch.save(saved_snapshots, snap_path)
    log.info(
        "Collected %d expert_srmp episodes (%d total steps, %d/%d succeeded, %d snapshots) -> %s",
        writer.num_episodes,
        writer.total_steps,
        success_count,
        collected,
        len(saved_snapshots),
        zarr_path,
    )
    return zarr_path, snap_path


# --------------------------------------------------------------------------- #
# Dataset 2: GPU-parallel state-branched exploratory collector
# --------------------------------------------------------------------------- #


def _sample_ou_noise(
    prev: torch.Tensor, theta: float, sigma: float, generator: torch.Generator
) -> torch.Tensor:
    rnd = torch.randn(prev.shape, device=prev.device, dtype=prev.dtype, generator=generator)
    return (1.0 - theta) * prev + sigma * rnd


def collect_branched_exploratory_dataset(
    cfg: DictConfig, snapshots_path: str | Path | None = None
) -> Path:
    """Collect GPU-parallel exploratory rollouts branched from saved SRMP snapshots."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_path = Path(snapshots_path or (out_dir / f"{cfg.task_id}_expert_srmp_snapshots.pt"))
    if not snap_path.is_file():
        raise FileNotFoundError(f"Snapshots file not found: {snap_path}")

    snapshots: list[dict[str, Any]] = torch.load(snap_path, weights_only=False)
    if not snapshots:
        raise RuntimeError(f"Snapshots file {snap_path} is empty")

    grasped_snaps = [s for s in snapshots if s.get("is_grasped", False)] or snapshots
    early_snaps = [s for s in snapshots if s["stage"] in {"approach", "grasp"}] or snapshots

    zarr_path = out_dir / f"{cfg.task_id}_branched_exploratory.zarr"
    bounds = _bounds_from_cfg(cfg)
    num_envs = int(cfg.num_envs)
    max_steps = int(cfg.max_episode_steps)

    env = gym.make(
        cfg.task_id,
        num_envs=num_envs,
        obs_mode="pointcloud",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cuda",
        render_mode=None,
        max_episode_steps=max_steps,
    )
    writer = ZarrTrajectoryWriter(
        zarr_path,
        num_points=int(cfg.num_points),
        action_dim=int(cfg.action_dim),
        attrs={
            "task_id": str(cfg.task_id),
            "control_mode": str(cfg.control_mode),
            "action_dim": int(cfg.action_dim),
            "dataset_type": "branched_exploratory",
            "workspace_bounds": [list(b) for b in bounds],
            "xyz_dtype": str(cfg.get("xyz_dtype", "float32")),
        },
        xyz_dtype=str(cfg.get("xyz_dtype", "float32")),
    )

    device = env.unwrapped.device
    rng_np = np.random.default_rng(int(cfg.seed) + 1000)
    gen = torch.Generator(device=device).manual_seed(int(cfg.seed) + 1000)

    p_noisy = float(cfg.branch_mixture.noisy_expert)
    p_inter = float(cfg.branch_mixture.interrupted)
    p_rand = float(cfg.branch_mixture.random_ou)
    probs = np.array([p_noisy, p_inter, p_rand], dtype=np.float64)
    probs /= probs.sum()

    sigma_lo, sigma_hi = float(cfg.expert_noise_sigma_range[0]), float(cfg.expert_noise_sigma_range[1])
    ou_theta = float(cfg.ou_theta)
    ou_sigma = float(cfg.ou_sigma)
    target_eps = int(cfg.branched_episodes)
    collected = 0
    batch_idx = 0

    pbar = trange(target_eps, desc=f"branched_exploratory ({cfg.task_id})")
    min_free_gb = float(cfg.get("min_free_gb", 0.0))
    while collected < target_eps:
        if not _disk_space_ok(out_dir, min_free_gb):
            break
        env.reset(seed=int(cfg.seed) + 5000 + batch_idx)
        batch_idx += 1

        # Sample trajectory categories per env: 0=noisy_expert, 1=interrupted, 2=random_ou.
        cats = rng_np.choice(3, size=num_envs, p=probs)
        inter_modes = rng_np.integers(0, 3, size=num_envs)  # 0: exp->rand, 1: held OU, 2: release/drop

        chosen_snaps: list[dict[str, Any]] = []
        for b in range(num_envs):
            if cats[b] == 0:
                chosen_snaps.append(early_snaps[int(rng_np.integers(len(early_snaps)))])
            elif cats[b] == 1 and inter_modes[b] in (1, 2):
                chosen_snaps.append(grasped_snaps[int(rng_np.integers(len(grasped_snaps)))])
            elif cats[b] == 1:
                chosen_snaps.append(snapshots[int(rng_np.integers(len(snapshots)))])
            else:
                # 50% full-episode OU from reset state, 50% OU from a random snapshot
                chosen_snaps.append(snapshots[int(rng_np.integers(len(snapshots)))])

        # Restore snapshots for non-reset trajectories (or all except random-from-reset subset).
        restore_mask = np.ones(num_envs, dtype=bool)
        for b in range(num_envs):
            if cats[b] == 2 and rng_np.random() < 0.5:
                restore_mask[b] = False

        if np.any(restore_mask):
            idx_list = np.where(restore_mask)[0].tolist()
            sub_state_dicts = [chosen_snaps[i]["state_dict"] for i in idx_list]
            env_idx_t = torch.tensor(idx_list, device=device, dtype=torch.int64)
            restore_snapshots_with_pd_sync(env, _stack_state_dicts(sub_state_dicts, device), env_idx=env_idx_t)

        # Refresh observations after state restoration!
        obs = env.unwrapped.get_obs()

        # Build padded expert waypoint tensors for all envs.
        max_wp_len = max(len(s["remaining_waypoints"]) for s in chosen_snaps)
        wp_tensor = torch.zeros((num_envs, max_wp_len, 7), device=device, dtype=torch.float32)
        grip_tensor = torch.ones((num_envs, max_wp_len), device=device, dtype=torch.float32)
        wp_lens = torch.zeros((num_envs,), device=device, dtype=torch.int64)
        for b, s in enumerate(chosen_snaps):
            L = len(s["remaining_waypoints"])
            wp_tensor[b, :L] = torch.from_numpy(s["remaining_waypoints"]).to(device)
            grip_tensor[b, :L] = torch.from_numpy(s["remaining_grippers"]).to(device)
            wp_lens[b] = L

        sigmas = torch.empty((num_envs, 1), device=device).uniform_(sigma_lo, sigma_hi, generator=gen)
        split_steps = rng_np.integers(max(4, int(0.2 * max_steps)), max(6, int(0.8 * max_steps)) + 1, size=num_envs)
        release_steps = rng_np.integers(3, max(6, max_steps // 3), size=num_envs)
        ou_state = torch.zeros((num_envs, 7), device=device, dtype=torch.float32)

        buf_xyz: list[np.ndarray] = []
        buf_rgb: list[np.ndarray] = []
        buf_act: list[np.ndarray] = []
        buf_prop: list[np.ndarray] = []
        buf_ach: list[np.ndarray] = []
        buf_ee: list[np.ndarray] = []

        for t in range(max_steps):
            xyz_b, rgb_b = preprocess_pointcloud_batch(
                obs["pointcloud"]["xyzw"],
                obs["pointcloud"]["rgb"],
                env.unwrapped.agent.robot.pose,
                bounds=bounds,
                num_points=int(cfg.num_points),
                voxel_size=float(cfg.voxel_size),
                generator=gen,
                sort_by_voxel=bool(cfg.get("sort_by_voxel", False)),
            )
            meta = extract_step_metadata(env, cfg.task_id)

            q_curr = env.unwrapped.agent.robot.get_qpos()[:, :7]
            wp_idx = torch.clamp(torch.full((num_envs,), t, device=device, dtype=torch.int64), max=wp_lens - 1)
            q_targ = wp_tensor[torch.arange(num_envs, device=device), wp_idx]
            g_targ = grip_tensor[torch.arange(num_envs, device=device), wp_idx]
            exp_action = joint_target_to_delta_action(q_curr, q_targ, g_targ, float(cfg.arm_delta_scale))

            ou_state = _sample_ou_noise(ou_state, ou_theta, ou_sigma, gen)
            unif_arm = torch.empty((num_envs, 7), device=device).uniform_(-1.0, 1.0, generator=gen)
            unif_grip = torch.where(
                torch.rand((num_envs, 1), device=device, generator=gen) > 0.5,
                torch.ones((num_envs, 1), device=device),
                -torch.ones((num_envs, 1), device=device),
            )
            gauss_arm = torch.randn((num_envs, 7), device=device, generator=gen) * sigmas

            actions = torch.zeros((num_envs, 8), device=device, dtype=torch.float32)
            for b in range(num_envs):
                cat = int(cats[b])
                if cat == 0:
                    actions[b, :7] = torch.clamp(exp_action[b, :7] + gauss_arm[b], -1.0, 1.0)
                    actions[b, 7] = exp_action[b, 7]
                elif cat == 1:
                    mode_i = int(inter_modes[b])
                    if mode_i == 0:
                        if t < int(split_steps[b]):
                            actions[b, :7] = torch.clamp(exp_action[b, :7] + gauss_arm[b], -1.0, 1.0)
                            actions[b, 7] = exp_action[b, 7]
                        else:
                            actions[b, :7] = unif_arm[b]
                            actions[b, 7] = unif_grip[b, 0]
                    elif mode_i == 1:
                        # Mid-grasp gripper-held OU motion
                        actions[b, :7] = torch.clamp(ou_state[b], -1.0, 1.0)
                        actions[b, 7] = -1.0
                    else:
                        # Release / drop motion during held transport
                        actions[b, :7] = torch.clamp(ou_state[b] * 0.5, -1.0, 1.0)
                        actions[b, 7] = -1.0 if t < int(release_steps[b]) else 1.0
                else:
                    actions[b, :7] = torch.clamp(ou_state[b], -1.0, 1.0)
                    actions[b, 7] = unif_grip[b, 0]

            buf_xyz.append(xyz_b.cpu().numpy())
            buf_rgb.append(rgb_b.cpu().numpy())
            buf_act.append(actions.cpu().numpy())
            buf_prop.append(meta["proprio"].cpu().numpy())
            buf_ach.append(meta["achieved_point"].cpu().numpy())
            buf_ee.append(meta["ee_point"].cpu().numpy())

            obs, _, _, _, _ = env.step(actions)

        # Transpose [T, B, ...] -> write each env's clean episode up to target_eps.
        arr_xyz = np.stack(buf_xyz, axis=1)
        arr_rgb = np.stack(buf_rgb, axis=1)
        arr_act = np.stack(buf_act, axis=1)
        arr_prop = np.stack(buf_prop, axis=1)
        arr_ach = np.stack(buf_ach, axis=1)
        arr_ee = np.stack(buf_ee, axis=1)

        n_write = min(num_envs, target_eps - collected)
        for b in range(n_write):
            writer.append_episode(
                arr_xyz[b], arr_rgb[b], arr_act[b], arr_prop[b], arr_ach[b], arr_ee[b]
            )
        collected += n_write
        pbar.update(n_write)

    pbar.close()
    env.close()
    log.info(
        "Collected %d branched_exploratory episodes (%d total steps) -> %s",
        writer.num_episodes,
        writer.total_steps,
        zarr_path,
    )
    return zarr_path


def export_sample_plys(zarr_path: str | Path, out_dir: str | Path, count: int = 2) -> list[Path]:
    """Export ``count`` frames from a Zarr dataset to colored ``.ply`` files."""
    if count <= 0:
        return []
    root = zarr.open_group(str(zarr_path), mode="r")
    n_steps = int(root["xyz"].shape[0])
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, max(0, n_steps - 1), num=min(count, n_steps), dtype=int)
    written: list[Path] = []
    for i, idx in enumerate(indices):
        xyz = np.asarray(root["xyz"][int(idx)], dtype=np.float32)
        rgb = np.asarray(root["rgb"][int(idx)], dtype=np.uint8)
        alpha = np.full((xyz.shape[0], 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgb, alpha], axis=-1)
        cloud = trimesh.PointCloud(vertices=xyz, colors=rgba)
        ply_file = out_path / f"{Path(zarr_path).stem}_frame_{i}_step_{int(idx)}.ply"
        cloud.export(str(ply_file))
        written.append(ply_file)
    return written


@hydra.main(config_path="../../configs", config_name="collect_maniskill", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("Collector configuration:\n%s", OmegaConf.to_yaml(cfg))
    mode = str(cfg.mode)
    snap_path = cfg.snapshots_path
    zarr_for_ply: Path | None = None

    if mode in ("expert_srmp", "both"):
        exp_zarr, snap_path = collect_expert_srmp_dataset(cfg)
        zarr_for_ply = exp_zarr

    if mode in ("branched_exploratory", "both"):
        br_zarr = collect_branched_exploratory_dataset(cfg, snapshots_path=snap_path)
        zarr_for_ply = br_zarr

    if mode not in ("expert_srmp", "branched_exploratory", "both"):
        raise ValueError(f"Unknown mode {mode!r}; choose 'expert_srmp', 'branched_exploratory', or 'both'")

    if zarr_for_ply is not None and int(cfg.export_ply_count) > 0:
        plys = export_sample_plys(zarr_for_ply, Path(cfg.output_dir) / "ply_samples", int(cfg.export_ply_count))
        log.info("Exported sample PLY files: %s", [str(p) for p in plys])


if __name__ == "__main__":
    main()
