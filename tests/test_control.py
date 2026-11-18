import numpy as np

from point_hjepa.data.control import apply_pd_ee_scaling, normalized_ee_delta


def test_delta_action_inverts_maniskill_scaling():
    # rot_lower is negative in ManiSkill 3.0.1, which flips the Euler command.
    delta_p = np.array([0.05, -0.02, 0.0])
    delta_e = np.array([0.0, 0.0, 0.04])
    action = normalized_ee_delta(
        delta_p, delta_e, gripper=-1.0, pos_lower=-0.1, pos_upper=0.1, rot_lower=-0.1
    )
    pos, rot = apply_pd_ee_scaling(action, pos_lower=-0.1, pos_upper=0.1, rot_lower=-0.1)
    assert np.allclose(pos, delta_p)
    assert np.allclose(rot, delta_e)
    assert action.shape == (7,)
    assert action[-1] == -1.0
    assert np.all(np.abs(action[:6]) <= 1.0)
