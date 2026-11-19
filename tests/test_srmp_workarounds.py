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


def test_plan_screw_reaches_after_set_base_pose():
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
    # 0.2.2 missed by about 0.19 m. 0.3.0 applies the base translation.
    assert moved_error < 0.02
