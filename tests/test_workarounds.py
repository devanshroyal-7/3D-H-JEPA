import numpy as np

from point_hjepa.data.geometry import world_to_base
from point_hjepa.data.planners.workarounds import (
    POST_GRASP_LIFT_M,
    compensate_screw_target,
    cutout_covers_base,
    is_ground_actor,
    is_table_actor,
    table_slabs_with_base_cutout,
)


def test_screw_bias_is_subtracted():
    desired = np.array([0.4, 0.1, 0.3])
    bias = np.array([0.2, 0.0, 0.0])
    assert np.allclose(compensate_screw_target(desired, bias), [0.2, 0.1, 0.3])


def test_table_cutout_leaves_the_base_clear():
    slabs = table_slabs_with_base_cutout(
        np.array([-0.5, -1.0, -0.9]),
        np.array([1.0, 1.0, 0.0]),
        cutout_half=0.18,
    )
    assert slabs
    assert cutout_covers_base(slabs, half=0.18)
    # The slab is the top skin, not the full 0.9 m table that sits flush with the base.
    for _center, size in slabs:
        assert size[2] < 0.05


def test_actor_filters_and_lift_constant():
    assert is_table_actor("scene-0_table-workspace")
    assert is_ground_actor("scene-0_ground")
    assert not is_table_actor("cube")
    assert POST_GRASP_LIFT_M == 0.05


def test_base_frame_cancels_the_maniskill_mount():
    # PickCube mounts the Panda at x = -0.615. A cube at the world origin is
    # 0.615 m in front of the base, which is the frame we plan in.
    p, q = world_to_base(
        np.array([0.0, 0.05, 0.02]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([-0.615, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    assert np.allclose(p, [0.615, 0.05, 0.02])
    assert np.allclose(q, [1.0, 0.0, 0.0, 0.0])
