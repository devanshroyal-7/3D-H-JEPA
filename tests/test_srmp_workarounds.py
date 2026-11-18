"""Live checks of the SRMP issues the collector works around.

These import srmp / ManiSkill and are skipped when those packages are absent.
"""

import numpy as np
import pytest

srmp = pytest.importorskip("srmp")


def _pose(p, q):
    pose = srmp.Pose()
    pose.p = np.asarray(p, dtype=np.float64)
    pose.q = np.asarray(q, dtype=np.float64)
    return pose


def test_plan_screw_misses_after_set_base_pose_and_base_frame_does_not():
    from point_hjepa.data.geometry import world_to_base

    planner = srmp.PlannerInterface()
    art = planner.add_robot("panda")
    home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    limits = np.asarray(planner.get_move_group_joint_limits(art), dtype=np.float64)
    q2 = np.clip(
        home + np.array([0.3, 0.2, -0.2, 0.3, -0.2, 0.2, 0.1]),
        limits[:, 0] + 1e-3,
        limits[:, 1] - 1e-3,
    )
    base_p = np.array([-0.615, 0.0, 0.0])
    base_q = np.array([1.0, 0.0, 0.0, 0.0])
    planner.set_base_pose(art, _pose(base_p, base_q))
    planner.set_qpos(art, q2)
    target = planner.compute_fk(art, q2)
    target_p = np.array(target.p, dtype=np.float64).copy()
    target_q = np.array(target.q, dtype=np.float64).copy()
    traj = planner.plan_screw(
        art,
        "panda_hand",
        start_qpos=home,
        end_pose=_pose(target_p, target_q),
        collision_aware=False,
        qpos_step=0.08,
        max_steps=800,
    )
    achieved = planner.compute_fk(art, np.asarray(traj.positions)[-1])
    moved_error = float(np.linalg.norm(np.array(achieved.p) - target_p))
    assert moved_error > 0.10

    planner.set_base_pose(art, _pose(np.zeros(3), base_q))
    target_pb, target_qb = world_to_base(target_p, target_q, base_p, base_q)
    traj = planner.plan_screw(
        art,
        "panda_hand",
        start_qpos=home,
        end_pose=_pose(target_pb, target_qb),
        collision_aware=False,
        qpos_step=0.08,
        max_steps=800,
    )
    achieved = planner.compute_fk(art, np.asarray(traj.positions)[-1])
    fixed_error = float(np.linalg.norm(np.array(achieved.p) - target_pb))
    assert fixed_error < 0.03
    assert moved_error > fixed_error * 5
