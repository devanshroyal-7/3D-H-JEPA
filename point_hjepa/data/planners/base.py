"""Planner interface shared by the SRMP default and the mplib fallback."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PlanResult:
    success: bool
    message: str
    # TCP poses in the world frame along the planned motion, including the goal.
    tcp_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    tcp_quats: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float64))
    joint_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 7), dtype=np.float64))


class MotionPlanner:
    """Plans Panda end-effector motions for one ManiSkill CPU scene.

    Implementations plan in the robot base frame. Callers pass world-frame TCP
    poses; the backend converts them.
    """

    name: str = "abstract"

    def setup(self, env) -> None:
        raise NotImplementedError

    def sync(self, env) -> None:
        """Refresh joint state and collision objects from the simulator."""
        raise NotImplementedError

    def plan_tcp(
        self,
        env,
        target_p: np.ndarray,
        target_q: np.ndarray,
        *,
        collision_aware: bool = True,
    ) -> PlanResult:
        raise NotImplementedError

    def set_gripper(self, env, closed: bool) -> None:
        raise NotImplementedError

    def ignore_object(self, attr_name: str) -> None:
        raise NotImplementedError

    def attach(self, env, attr_name: str) -> bool:
        raise NotImplementedError

    def detach_all(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None

    @property
    def base_position(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def base_quat(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def tcp_offset_in_hand(self) -> np.ndarray:
        raise NotImplementedError
