"""Scripted experts. SRMP plans the end-effector goals; the env is driven with pd_ee_delta_pose.

The 5 cm post-grasp lift stays collision-unaware. On srmp 0.3.0 a grasp 5 mm
off the table still penetrates the table by about 4 cm in coal, so a
collision-checked plan cannot start until the cube has been lifted. The older
mismatch (coal free, A* start invalid) did not reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from point_hjepa.data.control import CLOSED, OPEN, action_toward_tcp, hold_action, pose_error
from point_hjepa.data.geometry import quat4, vec3
from point_hjepa.data.planners.workarounds import POST_GRASP_LIFT_M

TargetFn = Callable[[object], tuple[np.ndarray, np.ndarray]]
Hook = Callable[[object, object], None]


@dataclass
class Phase:
    name: str
    kind: str
    steps: int
    gripper: float
    collision_aware: bool = True
    pos_tol: float = 0.03
    rot_tol: float = 0.45
    target_fn: TargetFn | None = None
    on_start: Hook | None = None


def _scales(env) -> dict[str, float]:
    controller = env.unwrapped.agent.controller.controllers["arm"]
    return {
        "pos_lower": float(np.asarray(controller.config.pos_lower).reshape(-1)[0]),
        "pos_upper": float(np.asarray(controller.config.pos_upper).reshape(-1)[0]),
        "rot_lower": float(np.asarray(controller.config.rot_lower).reshape(-1)[0]),
    }


def _tcp(env) -> tuple[np.ndarray, np.ndarray]:
    pose = env.unwrapped.agent.tcp.pose
    return vec3(pose.p), quat4(pose.q)


def _above(point: np.ndarray, quat: np.ndarray, height: float) -> tuple[np.ndarray, np.ndarray]:
    lifted = np.array(point, dtype=np.float64, copy=True)
    lifted[2] = lifted[2] + height
    return lifted, quat


class ExpertPolicy:
    def __init__(self, task_expert: str, planner, verbose: bool = False) -> None:
        self.task_expert = task_expert
        self.planner = planner
        self.verbose = verbose
        self.phases: list[Phase] = []
        self.index = 0
        self.used = 0
        self.waypoints: list[tuple[np.ndarray, np.ndarray]] = []
        self.waypoint = 0
        self.gripper = OPEN
        self.scales: dict[str, float] = {}
        self.hold_q = np.array([1.0, 0.0, 0.0, 0.0])

    def reset(self, env) -> None:
        self.scales = _scales(env)
        self.planner.sync(env)
        self.hold_q = quat4(env.unwrapped.agent.tcp.pose.q)
        self.phases = build_phases(self.task_expert, self.hold_q)
        self.index = -1
        self._enter(env)

    def act(self, env) -> np.ndarray:
        if self.index >= len(self.phases):
            return hold_action(self.gripper)
        phase = self.phases[self.index]
        if phase.kind == "gripper":
            action = hold_action(phase.gripper)
            self.gripper = phase.gripper
            self.used += 1
            if self.used >= phase.steps:
                self._enter(env)
            return action
        if self.waypoint >= len(self.waypoints):
            self._enter(env)
            return hold_action(self.gripper)
        target_p, target_q = self.waypoints[self.waypoint]
        tcp_p, tcp_q = _tcp(env)
        action = action_toward_tcp(
            tcp_p,
            tcp_q,
            target_p,
            target_q,
            self.planner.base_position,
            self.planner.base_quat,
            phase.gripper,
            **self.scales,
        )
        self.used += 1
        pos_err, rot_err = pose_error(tcp_p, tcp_q, target_p, target_q)
        share = max(1, phase.steps // max(1, len(self.waypoints)))
        reached = pos_err < phase.pos_tol and rot_err < phase.rot_tol
        overtime = self.used >= share * (self.waypoint + 1) or self.used >= phase.steps
        if reached or overtime:
            self.waypoint += 1
            if self.waypoint >= len(self.waypoints) or self.used >= phase.steps:
                self._enter(env)
        return action

    def _enter(self, env) -> None:
        self.index += 1
        self.used = 0
        self.waypoint = 0
        self.waypoints = []
        if self.index >= len(self.phases):
            return
        phase = self.phases[self.index]
        self.gripper = phase.gripper
        if self.verbose:
            print(f"  phase {phase.name} ({phase.kind})", flush=True)
        if phase.on_start is not None:
            phase.on_start(env, self.planner)
        if phase.kind == "gripper" or phase.target_fn is None:
            return
        target_p, target_q = phase.target_fn(env)
        if phase.kind == "planned":
            self.planner.set_gripper(env, phase.gripper < 0)
            result = self.planner.plan_tcp(
                env, target_p, target_q, collision_aware=phase.collision_aware
            )
            if result.success and len(result.tcp_positions):
                self.waypoints = list(zip(result.tcp_positions, result.tcp_quats))
            else:
                if self.verbose:
                    print(f"  plan failed ({result.message}); tracking the Cartesian target", flush=True)
                self.waypoints = [(target_p, target_q)]
        else:
            # Cartesian phases are the collision-off motions, including the
            # 5 cm post-grasp lift. They are not sent through plan_screw.
            self.waypoints = [(target_p, target_q)]


def build_phases(expert: str, hold_q: np.ndarray) -> list[Phase]:
    builders = {
        "pick": _pick_phases,
        "push": _push_phases,
        "stack": _stack_phases,
        "pull": _pull_phases,
        "peg": _peg_phases,
        "clutter": _clutter_phases,
    }
    if expert not in builders:
        raise KeyError(expert)
    return builders[expert](hold_q)


def _pick_phases(hold_q: np.ndarray) -> list[Phase]:
    def grasp(env, height: float):
        point = vec3(env.unwrapped.cube.pose.p)
        return _above(point, hold_q, height)

    def goal(env):
        return vec3(env.unwrapped.goal_site.pose.p), hold_q

    def lift(env):
        point, quat = _tcp(env)
        return _above(point, quat, POST_GRASP_LIFT_M)

    return [
        Phase("pregrasp", "planned", 10, OPEN, True, target_fn=lambda env: grasp(env, 0.08)),
        Phase("grasp", "planned", 8, OPEN, False, target_fn=lambda env: grasp(env, 0.0)),
        Phase("close", "gripper", 6, CLOSED),
        Phase("post_grasp_lift", "cartesian", 6, CLOSED, target_fn=lift),
        Phase(
            "transport",
            "planned",
            14,
            CLOSED,
            True,
            target_fn=goal,
            on_start=lambda env, planner: planner.attach(env, "cube"),
        ),
        Phase("settle", "gripper", 6, CLOSED),
    ]


def _push_phases(hold_q: np.ndarray) -> list[Phase]:
    def approach(env):
        obj = vec3(env.unwrapped.obj.pose.p)
        return obj + np.array([-0.07, 0.0, 0.0]), hold_q

    def push(env):
        obj = vec3(env.unwrapped.obj.pose.p)
        goal = vec3(env.unwrapped.goal_region.pose.p)
        return np.array([goal[0] - 0.10, goal[1], obj[2]]), hold_q

    return [
        Phase("close_gripper", "gripper", 4, CLOSED),
        Phase("approach", "planned", 14, CLOSED, True, target_fn=approach),
        Phase(
            "push",
            "planned",
            18,
            CLOSED,
            False,
            target_fn=push,
            on_start=lambda env, planner: planner.ignore_object("obj"),
        ),
        Phase("settle", "gripper", 6, CLOSED),
    ]


def _stack_phases(hold_q: np.ndarray) -> list[Phase]:
    def grasp(env, height: float):
        return _above(vec3(env.unwrapped.cubeA.pose.p), hold_q, height)

    def lift(env):
        point, quat = _tcp(env)
        return _above(point, quat, POST_GRASP_LIFT_M)

    def above(env, clearance: float):
        bottom = vec3(env.unwrapped.cubeB.pose.p)
        half = np.asarray(env.unwrapped.cube_half_size).reshape(-1)
        z_half = float(half[2] if half.size >= 3 else half[0])
        return np.array([bottom[0], bottom[1], bottom[2] + 2.0 * z_half + clearance]), hold_q

    return [
        Phase("pregrasp", "planned", 12, OPEN, True, target_fn=lambda env: grasp(env, 0.08)),
        Phase("grasp", "planned", 8, OPEN, False, target_fn=lambda env: grasp(env, 0.0)),
        Phase("close", "gripper", 6, CLOSED),
        Phase("post_grasp_lift", "cartesian", 8, CLOSED, target_fn=lift),
        Phase(
            "above",
            "planned",
            16,
            CLOSED,
            True,
            target_fn=lambda env: above(env, 0.06),
            on_start=lambda env, planner: planner.attach(env, "cubeA"),
        ),
        Phase("lower", "planned", 10, CLOSED, False, target_fn=lambda env: above(env, 0.005)),
        Phase("release", "gripper", 6, OPEN),
        Phase("retreat", "cartesian", 6, OPEN, target_fn=lift),
    ]


def _pull_phases(hold_q: np.ndarray) -> list[Phase]:
    def tool(env, height: float):
        return _above(vec3(env.unwrapped.l_shape_tool.pose.p), hold_q, height)

    def lift(env):
        point, quat = _tcp(env)
        return _above(point, quat, POST_GRASP_LIFT_M)

    def behind(env):
        cube = vec3(env.unwrapped.cube.pose.p)
        hook = float(getattr(env.unwrapped, "hook_length", 0.1))
        half = float(np.asarray(env.unwrapped.cube_half_size).reshape(-1)[0])
        return np.array([cube[0] - hook - half - 0.02, cube[1], cube[2] + 0.02]), hold_q

    def pull(env):
        point, quat = _tcp(env)
        return point + np.array([-0.18, 0.0, 0.0]), quat

    return [
        Phase("pregrasp", "planned", 12, OPEN, True, target_fn=lambda env: tool(env, 0.08)),
        Phase("grasp", "planned", 8, OPEN, False, target_fn=lambda env: tool(env, 0.0)),
        Phase("close", "gripper", 6, CLOSED),
        Phase("post_grasp_lift", "cartesian", 8, CLOSED, target_fn=lift),
        Phase(
            "behind",
            "planned",
            16,
            CLOSED,
            False,
            target_fn=behind,
            on_start=lambda env, planner: planner.attach(env, "l_shape_tool"),
        ),
        Phase("pull", "cartesian", 16, CLOSED, target_fn=pull),
        Phase("settle", "gripper", 6, CLOSED),
    ]


def _peg_phases(hold_q: np.ndarray) -> list[Phase]:
    def tail(env, height: float):
        unwrapped = env.unwrapped
        peg = vec3(unwrapped.peg.pose.p)
        half = float(np.asarray(unwrapped.peg_half_sizes).reshape(-1)[0])
        point = peg + np.array([-max(0.05, half * 0.5), 0.0, height])
        return point, hold_q

    def lift(env):
        point, quat = _tcp(env)
        return _above(point, quat, POST_GRASP_LIFT_M)

    def preinsert(env):
        pose = env.unwrapped.goal_pose
        point = vec3(pose.p) + np.array([-0.08, 0.0, 0.0])
        return point, quat4(pose.q)

    def insert(env):
        pose = env.unwrapped.goal_pose
        return vec3(pose.p) + np.array([0.02, 0.0, 0.0]), quat4(pose.q)

    return [
        Phase("pregrasp", "planned", 14, OPEN, True, target_fn=lambda env: tail(env, 0.08)),
        Phase("grasp", "planned", 10, OPEN, False, target_fn=lambda env: tail(env, 0.0)),
        Phase("close", "gripper", 6, CLOSED),
        Phase("post_grasp_lift", "cartesian", 8, CLOSED, target_fn=lift),
        Phase(
            "preinsert",
            "planned",
            16,
            CLOSED,
            True,
            target_fn=preinsert,
            on_start=lambda env, planner: planner.attach(env, "peg"),
        ),
        Phase("insert", "planned", 12, CLOSED, False, target_fn=insert),
        Phase("settle", "gripper", 6, CLOSED),
    ]


def _clutter_phases(hold_q: np.ndarray) -> list[Phase]:
    def grasp(env, height: float):
        return _above(vec3(env.unwrapped.target_object.pose.p), hold_q, height)

    def goal(env):
        return vec3(env.unwrapped.goal_pos), hold_q

    def lift(env):
        point, quat = _tcp(env)
        return _above(point, quat, POST_GRASP_LIFT_M)

    return [
        Phase("pregrasp", "planned", 12, OPEN, True, target_fn=lambda env: grasp(env, 0.08)),
        Phase("grasp", "planned", 10, OPEN, False, target_fn=lambda env: grasp(env, 0.0)),
        Phase("close", "gripper", 6, CLOSED),
        Phase("post_grasp_lift", "cartesian", 8, CLOSED, target_fn=lift),
        Phase(
            "transport",
            "planned",
            16,
            CLOSED,
            False,
            target_fn=goal,
            on_start=lambda env, planner: planner.attach(env, "target_object"),
        ),
        Phase("settle", "gripper", 6, CLOSED),
    ]
