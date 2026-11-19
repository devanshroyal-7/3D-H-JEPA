"""SRMP backend (default).

On srmp 0.3.0 the planner base is the robot base (``set_base_pose``) and
targets are world-frame poses. Static geometry comes from ``read_sim``.
``set_base_contact_allowed`` lets the fixed base rest on the imported table.
Search uses coal, because VAMP still rejects every state when those base
spheres overlap the table. See ``workarounds.py``.
"""

from __future__ import annotations

import logging

import numpy as np

from point_hjepa.data.geometry import (
    hand_pose_from_tcp,
    quat4,
    tcp_pose_from_hand,
    vec3,
)
from point_hjepa.data.planners.base import MotionPlanner, PlanResult
from point_hjepa.data.planners.workarounds import POST_GRASP_LIFT_M, is_ground_actor, is_table_actor

logger = logging.getLogger(__name__)

_IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


class SrmpPlanner(MotionPlanner):
    name = "srmp"

    def __init__(self) -> None:
        self._planner = None
        self.art_name = "panda"
        self._ee_link = "panda_hand"
        self._base_p = np.zeros(3, dtype=np.float64)
        self._base_q = _IDENTITY_Q.copy()
        self._tcp_offset = np.array([0.0, 0.0, -0.1034], dtype=np.float64)
        self._attr_to_name: dict[str, str] = {}
        self._ignored: set[str] = set()
        self._attached: set[str] = set()
        self._object_names: list[str] = []
        self._imported_names: list[str] = []
        self._ready = False
        self._current_arm_qpos: np.ndarray | None = None
        self.plan_failures = 0
        # Overridden per task. 0.03 m on large meshes, else srmp's 0.01 m.
        self.mesh_sampling_resolution = 0.01
        self.diagnostics: dict = {
            "planner": "srmp",
            "used_set_base_pose": True,
            "base_contact_allowed": True,
            "validity_backend": "coal",
            "post_grasp_lift_m": POST_GRASP_LIFT_M,
            "mesh_sampling_resolution": self.mesh_sampling_resolution,
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
        if self._ready:
            return
        import srmp

        grid = srmp.GridConfig()
        # The default grid starts at z = 0, so the floor (about 0.92 m below
        # the table) would be outside the distance field.
        grid.origin_x = -1.5
        grid.origin_y = -1.5
        grid.origin_z = -1.2
        grid.size_x = 3.0
        grid.size_y = 3.0
        grid.size_z = 2.6
        grid.resolution = 0.025
        self._planner = srmp.PlannerInterface(grid_config=grid)
        self.art_name = self._planner.add_robot("panda")
        links = list(self._planner.get_link_names(self.art_name))
        self._ee_link = "panda_hand" if "panda_hand" in links else links[-1]
        self._measure_tcp_offset(env)
        self._ready = True

    def sync(self, env) -> None:
        self.setup(env)
        self.detach_all()
        self._clear_objects()
        self._ignored.clear()
        self._read_base(env)
        self._planner.set_base_pose(self.art_name, self._pose(self._base_p, self._base_q))
        # Fixed base links may rest on the imported table. Must be set before
        # read_sim, which otherwise warns that every state is in collision.
        self._planner.set_base_contact_allowed(self.art_name, True)
        self._import_static_scene(env)
        self._add_task_objects(env)
        self._set_robot_state(env)
        self.diagnostics["mesh_sampling_resolution"] = float(self.mesh_sampling_resolution)
        self.diagnostics["start_colliding"] = bool(self._planner.is_state_colliding(self.art_name))

    def plan_tcp(self, env, target_p: np.ndarray, target_q: np.ndarray, *, collision_aware: bool = True) -> PlanResult:
        self._set_robot_state(env)
        self._refresh_dynamic_objects(env)
        hand_p, hand_q = hand_pose_from_tcp(target_p, target_q, self._tcp_offset)
        qpos = self._current_arm_qpos
        traj, message, ok = self._screw(hand_p, hand_q, qpos, collision_aware)
        if not ok:
            traj, message, ok = self._search(hand_p, hand_q, qpos)
        if not ok or traj is None:
            self.plan_failures += 1
            return PlanResult(False, message)
        joints = np.asarray(traj.positions, dtype=np.float64)
        if joints.ndim == 1:
            joints = joints.reshape(1, -1)
        positions, quats = self._tcp_waypoints_world(joints, target_p, target_q)
        return PlanResult(True, message, positions, quats, joints[:, :7] if joints.shape[1] >= 7 else joints)

    def set_gripper(self, env, closed: bool) -> None:
        if self._planner is None:
            return
        opening = 0.0 if closed else 0.04
        self._planner.set_gripper_qpos(self.art_name, np.array([opening, opening], dtype=np.float64))

    def ignore_object(self, attr_name: str) -> None:
        name = self._attr_to_name.get(attr_name)
        if name and self._planner is not None and self._planner.has_object(name):
            if attr_name in self._attached:
                self._planner.detach_object(name)
                self._attached.discard(attr_name)
            self._planner.remove_object(name)
        self._ignored.add(attr_name)

    def attach(self, env, attr_name: str) -> bool:
        if self._planner is None or attr_name not in self._attr_to_name:
            return False
        if attr_name in self._ignored:
            self._ignored.discard(attr_name)
            self._readd_object(env, attr_name)
        else:
            self._update_object_pose(env, attr_name)
        name = self._attr_to_name[attr_name]
        try:
            self._planner.attach_object(name, self.art_name, self._ee_link, is_link_name=True)
            self._attached.add(attr_name)
            return True
        except Exception as exc:  # noqa: BLE001 - planner-specific failure is data, not a bug
            logger.warning("SRMP attach %s failed: %s", attr_name, exc)
            return False

    def detach_all(self) -> None:
        if self._planner is None:
            return
        try:
            self._planner.detach_all_objects(also_remove=False)
        except Exception:
            for attr in list(self._attached):
                name = self._attr_to_name.get(attr)
                if name:
                    try:
                        self._planner.detach_object(name)
                    except Exception:
                        pass
        self._attached.clear()

    def close(self) -> None:
        self._planner = None
        self._ready = False

    # --- internals ---

    def _read_base(self, env) -> None:
        robot = env.unwrapped.agent.robot
        self._base_p = vec3(robot.pose.p)
        self._base_q = quat4(robot.pose.q)

    def _measure_tcp_offset(self, env) -> None:
        robot = env.unwrapped.agent.robot
        hand = next(link for link in robot.get_links() if link.name == "panda_hand")
        hand_p, hand_q = vec3(hand.pose.p), quat4(hand.pose.q)
        tcp_p = vec3(env.unwrapped.agent.tcp.pose.p)
        from point_hjepa.data.geometry import quat_to_mat

        self._tcp_offset = quat_to_mat(hand_q).T @ (tcp_p - hand_p)
        self.diagnostics["tcp_offset_in_hand"] = [float(v) for v in self._tcp_offset]

    def _arm_qpos(self, env) -> np.ndarray:
        robot = env.unwrapped.agent.robot
        q_all = np.asarray(robot.get_qpos().detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        # Batched CPU envs still return (1, nq).
        if q_all.ndim > 1:
            q_all = q_all[0]
        joint_names = [joint.name for joint in robot.get_active_joints()]
        move = list(self._planner.get_move_group_joint_names(self.art_name))
        return np.array([q_all[joint_names.index(name)] for name in move], dtype=np.float64)

    def _set_robot_state(self, env, closed: bool | None = None) -> None:
        qpos = self._arm_qpos(env)
        self._current_arm_qpos = qpos
        self._planner.set_qpos(self.art_name, qpos)
        if closed is None:
            finger = np.asarray(env.unwrapped.agent.robot.get_qpos().detach().cpu().numpy(), dtype=np.float64).reshape(-1)
            closed = bool(finger.reshape(-1)[-1] < 0.02)
        self.set_gripper(env, closed)

    def _refresh_dynamic_objects(self, env) -> None:
        for attr in list(self._attr_to_name):
            if attr in self._ignored:
                continue
            was_attached = attr in self._attached
            name = self._attr_to_name[attr]
            if was_attached:
                try:
                    self._planner.detach_object(name)
                except Exception:
                    pass
            self._update_object_pose(env, attr)
            if was_attached and self._planner.has_object(name):
                try:
                    self._planner.attach_object(name, self.art_name, self._ee_link, is_link_name=True)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("re-attach %s failed: %s", attr, exc)
                    self._attached.discard(attr)

    def _clear_objects(self) -> None:
        for name in list(self._object_names) + list(self._imported_names):
            if self._planner.has_object(name):
                self._planner.remove_object(name)
        self._object_names.clear()
        self._imported_names.clear()
        self._attr_to_name.clear()

    def _drop_names_with_prefix(self, prefix: str) -> None:
        for name in list(self._object_names):
            if name.startswith(prefix) and self._planner.has_object(name):
                self._planner.remove_object(name)
                self._object_names.remove(name)

    def _add_box(self, name: str, center: np.ndarray, size: np.ndarray) -> None:
        import srmp

        pose = srmp.Pose()
        pose.p = np.asarray(center, dtype=np.float64)
        pose.q = _IDENTITY_Q.copy()
        size = np.maximum(np.asarray(size, dtype=np.float64), 1e-3)
        self._planner.add_box(name, size, pose)
        self._object_names.append(name)

    def _add_task_objects(self, env) -> None:
        attrs = getattr(env, "_point_hjepa_actor_attrs", ())
        for attr in attrs:
            actor = getattr(env.unwrapped, attr, None)
            if actor is None:
                continue
            bounds = _actor_bounds(actor)
            if bounds is None:
                continue
            center_w = 0.5 * (bounds[0] + bounds[1])
            size = np.maximum(bounds[1] - bounds[0], 0.01)
            name = f"obj_{attr}"
            self._add_box(name, center_w, size)
            self._attr_to_name[attr] = name

    def _import_static_scene(self, env) -> None:
        """Import the sim scene and drop the task actors we manage ourselves."""
        scene = env.unwrapped.scene.sub_scenes[0]
        try:
            imported = self._planner.read_sim(scene, "sapien", articulations=["panda"]) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("SRMP read_sim failed: %s", exc)
            self.diagnostics["read_sim_error"] = str(exc)
            return
        task_names = set()
        for attr in getattr(env, "_point_hjepa_actor_attrs", ()):
            actor = getattr(env.unwrapped, attr, None)
            if actor is not None and getattr(actor, "name", None):
                task_names.add(str(actor.name))
        for source, objects in imported.items():
            if any(name and name in source for name in task_names):
                for obj_name in objects:
                    if self._planner.has_object(obj_name):
                        self._planner.remove_object(obj_name)
                continue
            self._imported_names.extend(objects)
        self.diagnostics["imported_static"] = list(self._imported_names)

    def _object_center_world(self, env, attr: str) -> np.ndarray | None:
        actor = getattr(env.unwrapped, attr, None)
        if actor is None:
            return None
        bounds = _actor_bounds(actor)
        if bounds is None:
            return None
        return 0.5 * (bounds[0] + bounds[1])

    def _update_object_pose(self, env, attr: str) -> None:
        import srmp

        name = self._attr_to_name.get(attr)
        center = self._object_center_world(env, attr)
        if name is None or center is None or not self._planner.has_object(name):
            return
        pose = srmp.Pose()
        pose.p = center
        pose.q = _IDENTITY_Q.copy()
        self._planner.update_object_pose(name, pose)

    def _readd_object(self, env, attr: str) -> None:
        if attr in self._attr_to_name and self._planner.has_object(self._attr_to_name[attr]):
            self._update_object_pose(env, attr)
            return
        actor = getattr(env.unwrapped, attr, None)
        bounds = _actor_bounds(actor) if actor is not None else None
        if bounds is None:
            return
        center = self._object_center_world(env, attr)
        size = np.maximum(bounds[1] - bounds[0], 0.01)
        name = f"obj_{attr}"
        self._add_box(name, center, size)
        self._attr_to_name[attr] = name

    def _pose(self, p: np.ndarray, q: np.ndarray):
        import srmp

        pose = srmp.Pose()
        pose.p = np.asarray(p, dtype=np.float64)
        pose.q = quat4(q)
        return pose

    def _screw(self, hand_pb, hand_qb, qpos, collision_aware: bool):
        try:
            traj = self._planner.plan_screw(
                self.art_name,
                self._ee_link,
                start_qpos=np.asarray(qpos, dtype=np.float64),
                end_pose=self._pose(hand_pb, hand_qb),
                collision_aware=bool(collision_aware),
                qpos_step=0.08,
                max_steps=500,
            )
            positions = np.asarray(traj.positions)
            if positions.size == 0:
                return None, "plan_screw returned an empty trajectory", False
            return traj, "plan_screw", True
        except Exception as exc:  # noqa: BLE001
            return None, f"plan_screw failed: {exc}", False

    def _search(self, hand_pb, hand_qb, qpos):
        import srmp

        try:
            self._planner.make_planner([self.art_name], self._planner_context())
            goal = srmp.GoalConstraint(
                srmp.GoalType.POSE, poses=[self._pose(hand_pb, hand_qb)]
            )
            traj = self._planner.plan(np.asarray(qpos, dtype=np.float64), goal)
            positions = np.asarray(traj.positions)
            if positions.size == 0:
                return None, "srmp plan() returned an empty trajectory", False
            return traj, "plan", True
        except Exception as exc:  # noqa: BLE001
            return None, f"plan failed: {exc}", False

    def _planner_context(self) -> dict[str, str]:
        # coal honors set_base_contact_allowed. VAMP's base spheres still
        # overlap a table the base rests on, so every VAMP state is invalid.
        return {
            "planner_id": "wAstar",
            "heuristic": "bfs",
            "weight": "10",
            "time_limit": "2",
            "validity_backend": "coal",
            "mesh_sampling_resolution": f"{float(self.mesh_sampling_resolution):.4f}",
        }

    def _fk_position(self, qpos: np.ndarray) -> np.ndarray:
        pose = self._planner.compute_fk(self.art_name, np.asarray(qpos, dtype=np.float64).reshape(-1))
        return np.array(pose.p, dtype=np.float64).copy()

    def _fk_pose(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pose = self._planner.compute_fk(self.art_name, np.asarray(qpos, dtype=np.float64).reshape(-1))
        return np.array(pose.p, dtype=np.float64).copy(), np.array(pose.q, dtype=np.float64).copy()

    def _tcp_waypoints_world(self, joints: np.ndarray, target_p, target_q):
        count = len(joints)
        pick = np.unique(np.round(np.linspace(0, count - 1, num=min(4, count))).astype(int))
        positions = []
        quats = []
        for index in pick:
            hand_p, hand_q = self._fk_pose(joints[index])
            tcp_p, tcp_q = tcp_pose_from_hand(hand_p, hand_q, self._tcp_offset)
            positions.append(tcp_p)
            quats.append(tcp_q)
        positions.append(np.asarray(target_p, dtype=np.float64))
        quats.append(quat4(target_q))
        return np.stack(positions), np.stack(quats)


def _actor_bounds(actor) -> np.ndarray | None:
    try:
        mesh = actor.get_first_collision_mesh(to_world_frame=True)
    except Exception:
        return None
    if mesh is None:
        return None
    return np.asarray(mesh.bounds, dtype=np.float64)


def _aabb_corners(bounds: np.ndarray) -> np.ndarray:
    xs = bounds[:, 0]
    ys = bounds[:, 1]
    zs = bounds[:, 2]
    corners = []
    for x in xs:
        for y in ys:
            for z in zs:
                corners.append([x, y, z])
    return np.asarray(corners, dtype=np.float64)


def _find_table(env):
    unwrapped = env.unwrapped
    table_scene = getattr(unwrapped, "table_scene", None)
    if table_scene is not None and getattr(table_scene, "table", None) is not None:
        return table_scene.table
    scene_builder = getattr(unwrapped, "scene_builder", None)
    if scene_builder is not None and getattr(scene_builder, "table", None) is not None:
        return scene_builder.table
    actors = getattr(unwrapped.scene, "actors", {})
    for actor in actors.values():
        if is_table_actor(getattr(actor, "name", "")):
            return actor
    return None


def probe_known_issues(env) -> dict:
    """Record the srmp 0.3.0 screw error and whether base contact clears the table."""
    import srmp

    report: dict = {"post_grasp_lift_m": POST_GRASP_LIFT_M}
    robot = env.unwrapped.agent.robot
    base_p = vec3(robot.pose.p)
    base_q = quat4(robot.pose.q)
    q_all = np.asarray(robot.get_qpos().detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    joint_names = [joint.name for joint in robot.get_active_joints()]

    grid = srmp.GridConfig()
    grid.origin_x, grid.origin_y, grid.origin_z = -1.5, -1.5, -0.6
    grid.size_x = grid.size_y = 3.0
    grid.size_z = 2.2
    grid.resolution = 0.03
    planner = srmp.PlannerInterface(grid_config=grid)
    art = planner.add_robot("panda")
    move = list(planner.get_move_group_joint_names(art))
    home = np.array([q_all[joint_names.index(name)] for name in move], dtype=np.float64)
    limits = np.asarray(planner.get_move_group_joint_limits(art), dtype=np.float64)
    delta = np.array([0.25, 0.15, -0.15, 0.25, -0.15, 0.15, 0.1], dtype=np.float64)
    q2 = np.clip(home + delta[: len(home)], limits[:, 0] + 1e-3, limits[:, 1] - 1e-3)

    base = srmp.Pose()
    base.p = base_p
    base.q = base_q
    planner.set_base_pose(art, base)
    planner.set_qpos(art, q2)
    target = planner.compute_fk(art, q2)
    target_p = np.array(target.p, dtype=np.float64).copy()
    target_q = np.array(target.q, dtype=np.float64).copy()
    try:
        traj = planner.plan_screw(
            art,
            "panda_hand",
            start_qpos=home,
            end_pose=_copy_pose(srmp, target_p, target_q),
            collision_aware=False,
            qpos_step=0.08,
            max_steps=800,
        )
        achieved = planner.compute_fk(art, np.asarray(traj.positions)[-1])
        err = np.array(achieved.p, dtype=np.float64) - target_p
        report["set_base_pose_screw_error_m"] = float(np.linalg.norm(err))
        report["set_base_pose_screw_error_xyz"] = [float(v) for v in err]
    except Exception as exc:  # noqa: BLE001
        report["set_base_pose_screw_error"] = str(exc)

    probe = srmp.PlannerInterface(grid_config=grid)
    art2 = probe.add_robot("panda")
    probe.set_base_pose(art2, base)
    probe.set_base_contact_allowed(art2, True)
    probe.set_qpos(art2, home)
    try:
        probe.set_gripper_qpos(art2, np.array([0.04, 0.04]))
    except Exception:
        pass
    before = bool(probe.is_state_colliding(art2))
    scene = env.unwrapped.scene.sub_scenes[0]
    try:
        imported = probe.read_sim(scene, "sapien", articulations=["panda"])
    except Exception as exc:  # noqa: BLE001
        report["read_sim_error"] = str(exc)
        return report
    after = bool(probe.is_state_colliding(art2))
    report["read_sim_start_collision_before_import"] = before
    report["read_sim_start_collision_with_base_contact_allowed"] = after
    ground_names = []
    if isinstance(imported, dict):
        for actor_name, objects in imported.items():
            if is_ground_actor(actor_name):
                ground_names.extend(objects)
            elif not is_table_actor(actor_name):
                for obj_name in objects:
                    if probe.has_object(obj_name):
                        probe.remove_object(obj_name)
        for actor_name, objects in imported.items():
            if is_table_actor(actor_name):
                for obj_name in objects:
                    if probe.has_object(obj_name):
                        probe.remove_object(obj_name)
    probe.set_qpos(art2, home)
    report["ground_only_reset_collide"] = bool(probe.is_state_colliding(art2))
    try:
        report["ground_only_distance_m"] = float(probe.distance_to_robot_collision())
    except Exception as exc:  # noqa: BLE001
        report["ground_only_distance_error"] = str(exc)
    report["read_sim_imported_ground"] = ground_names
    report["uses_set_base_pose"] = True
    report["base_contact_allowed"] = True
    return report


def _copy_pose(srmp, p, q):
    pose = srmp.Pose()
    pose.p = np.asarray(p, dtype=np.float64)
    pose.q = quat4(q)
    return pose
