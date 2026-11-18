"""Optional mplib backend.

mplib 0.1.1 is what ``mani_skill`` pins. ManiSkill's own solver notes that
mplib does not reliably accept a robot base away from the origin, so this
backend — like the SRMP one — never calls ``set_base_pose`` and plans in the
robot base frame. Select it with ``planner: mplib``. It is not used unless
asked for; collection does not fall back to it when SRMP fails.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from point_hjepa.data.geometry import (
    base_to_world,
    hand_pose_from_tcp,
    quat4,
    quat_to_mat,
    tcp_pose_from_hand,
    vec3,
    world_to_base,
)
from point_hjepa.data.planners.base import MotionPlanner, PlanResult
from point_hjepa.data.planners.srmp_backend import _aabb_corners, _actor_bounds, _find_table
from point_hjepa.data.planners.workarounds import ground_slab, table_slabs_with_base_cutout

logger = logging.getLogger(__name__)
_IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])


def _panda_urdf() -> str:
    import mani_skill

    path = Path(mani_skill.__file__).parent / "assets" / "robots" / "panda" / "panda_v2.urdf"
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)


class MplibPlanner(MotionPlanner):
    name = "mplib"

    def __init__(self) -> None:
        self._planner = None
        self._move_group = "panda_hand_tcp"
        self._base_p = np.zeros(3)
        self._base_q = _IDENTITY_Q.copy()
        self._tcp_offset = np.zeros(3)
        self._names: list[str] = []
        self._ignored: set[str] = set()
        self._attached = False
        self.plan_failures = 0
        self._current_qpos = None
        self.diagnostics = {
            "planner": "mplib",
            "used_set_base_pose": False,
            "note": "Plans in the robot base frame. set_base_pose is not called.",
        }

    @property
    def base_position(self) -> np.ndarray:
        return self._base_p

    @property
    def base_quat(self) -> np.ndarray:
        return self._base_q

    @property
    def tcp_offset_in_hand(self) -> np.ndarray:
        return self._tcp_offset

    def setup(self, env) -> None:
        if self._planner is not None:
            return
        import mplib

        urdf = _panda_urdf()
        # Prefer the TCP link so delta-pose targets do not need a hand offset.
        move_group = "panda_hand_tcp"
        try:
            self._planner = mplib.Planner(urdf=urdf, move_group=move_group, srdf="")
        except Exception:
            move_group = "panda_hand"
            self._planner = mplib.Planner(urdf=urdf, move_group=move_group, srdf="")
        self._move_group = move_group
        if move_group == "panda_hand":
            self._measure_offset(env)
        self.diagnostics["move_group"] = move_group

    def sync(self, env) -> None:
        self.setup(env)
        self._clear()
        self._base_p = vec3(env.unwrapped.agent.robot.pose.p)
        self._base_q = quat4(env.unwrapped.agent.robot.pose.q)
        self._add_scene(env)
        self._current_qpos = self._arm_qpos(env)

    def plan_tcp(self, env, target_p, target_q, *, collision_aware: bool = True) -> PlanResult:
        self._current_qpos = self._arm_qpos(env)
        if self._move_group == "panda_hand_tcp":
            point, quat = np.asarray(target_p, dtype=np.float64), quat4(target_q)
        else:
            point, quat = hand_pose_from_tcp(target_p, target_q, self._tcp_offset)
        pb, qb = world_to_base(point, quat, self._base_p, self._base_q)
        goal = np.concatenate([pb, qb])
        try:
            result = self._planner.plan_screw(
                goal,
                self._current_qpos,
                qpos_step=0.08,
                time_step=0.1,
                use_point_cloud=False,
                use_attach=self._attached,
                wrt_world=False,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            self.plan_failures += 1
            return PlanResult(False, f"mplib plan_screw failed: {exc}")
        if not isinstance(result, dict) or result.get("status") != "Success":
            self.plan_failures += 1
            return PlanResult(False, f"mplib status {result.get('status') if isinstance(result, dict) else result}")
        joints = np.asarray(result["position"], dtype=np.float64)
        positions, quats = self._waypoints(joints, target_p, target_q)
        return PlanResult(True, "mplib plan_screw", positions, quats, joints)

    def set_gripper(self, env, closed: bool) -> None:
        return None

    def ignore_object(self, attr_name: str) -> None:
        name = f"obj_{attr_name}"
        self._planner.remove_normal_object(name)
        self._ignored.add(attr_name)

    def attach(self, env, attr_name: str) -> bool:
        # mplib's attached-box API replaces a single tool. Record the intent;
        # the screw is still planned for the arm, and the 5 cm lift has
        # already cleared the table before this is called.
        self._attached = True
        return True

    def detach_all(self) -> None:
        self._attached = False

    def close(self) -> None:
        self._planner = None

    def _measure_offset(self, env) -> None:
        robot = env.unwrapped.agent.robot
        hand = next(link for link in robot.get_links() if link.name == "panda_hand")
        rot = quat_to_mat(quat4(hand.pose.q))
        self._tcp_offset = rot.T @ (vec3(env.unwrapped.agent.tcp.pose.p) - vec3(hand.pose.p))

    def _arm_qpos(self, env) -> np.ndarray:
        q_all = np.asarray(env.unwrapped.agent.robot.get_qpos().detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        # mplib's panda chain is the 7 arm joints; fingers are not in the move group.
        return q_all[:7].copy()

    def _clear(self) -> None:
        for name in self._names:
            self._planner.remove_normal_object(name)
        self._names.clear()
        self._ignored.clear()

    def _add_box(self, name: str, center: np.ndarray, size: np.ndarray) -> None:
        import mplib.pymp.collision_detection.fcl as fcl

        box = fcl.Box(np.maximum(size, 1e-3).tolist())
        obj = fcl.CollisionObject(box)
        transform = np.eye(4)
        transform[:3, 3] = center
        obj.set_transformation(transform)
        self._planner.set_normal_object(name, obj)
        self._names.append(name)

    def _add_scene(self, env) -> None:
        for attr in getattr(env, "_point_hjepa_actor_attrs", ()):
            actor = getattr(env.unwrapped, attr, None)
            bounds = _actor_bounds(actor) if actor is not None else None
            if bounds is None:
                continue
            center_w = 0.5 * (bounds[0] + bounds[1])
            size = np.maximum(bounds[1] - bounds[0], 0.01)
            center_b, _ = world_to_base(center_w, _IDENTITY_Q, self._base_p, self._base_q)
            self._add_box(f"obj_{attr}", center_b, size)
        table = _find_table(env)
        bounds = _actor_bounds(table) if table is not None else None
        if bounds is None:
            return
        corners = [
            world_to_base(c, _IDENTITY_Q, self._base_p, self._base_q)[0] for c in _aabb_corners(bounds)
        ]
        stacked = np.stack(corners)
        mins, maxs = stacked.min(0), stacked.max(0)
        for i, (center, size) in enumerate(table_slabs_with_base_cutout(mins, maxs)):
            self._add_box(f"mplib_table_{i}", center, size)
        center, size = ground_slab(mins[:2], maxs[:2])
        self._add_box("mplib_ground", center, size)

    def _waypoints(self, joints: np.ndarray, target_p, target_q):
        link = self._planner.link_name_2_idx[self._move_group]
        count = len(joints)
        pick = np.unique(np.round(np.linspace(0, count - 1, num=min(4, count))).astype(int))
        positions, quats = [], []
        for index in pick:
            qpos = self._planner.pad_qpos(joints[index].copy())
            self._planner.pinocchio_model.compute_forward_kinematics(qpos)
            pose7 = np.asarray(self._planner.pinocchio_model.get_link_pose(link), dtype=np.float64)
            point_b, quat_b = pose7[:3], pose7[3:]
            point_w, quat_w = base_to_world(point_b, quat_b, self._base_p, self._base_q)
            if self._move_group == "panda_hand":
                point_w, quat_w = tcp_pose_from_hand(point_w, quat_w, self._tcp_offset)
            positions.append(point_w)
            quats.append(quat4(quat_w))
        positions.append(np.asarray(target_p, dtype=np.float64))
        quats.append(quat4(target_q))
        return np.stack(positions), np.stack(quats)
