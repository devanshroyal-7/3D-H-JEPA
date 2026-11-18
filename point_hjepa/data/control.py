"""Map a desired end-effector delta onto ManiSkill `pd_ee_delta_pose` actions.

The Panda controller (`PDEEPoseController`) normalizes inputs to [-1, 1].
Translation is scaled into [`pos_lower`, `pos_upper`] (default ±0.1 m) in the
robot-root frame. Rotation is an XYZ Euler delta whose vector is clipped to
unit norm and then multiplied by `rot_lower`. In ManiSkill 3.0.1 `rot_lower`
is -0.1, so the sign of the rotation command is flipped relative to the
physical Euler delta. `normalized_ee_delta` inverts that mapping.
"""

from __future__ import annotations

import numpy as np

from point_hjepa.data.geometry import quat4, quat_inv, quat_mul, quat_to_euler_xyz, world_to_base

OPEN = 1.0
CLOSED = -1.0


def unscale_interval(value: np.ndarray, low: float, high: float) -> np.ndarray:
    span = float(high) - float(low)
    if abs(span) < 1e-12:
        return np.zeros_like(value, dtype=np.float64)
    return (2.0 * np.asarray(value, dtype=np.float64) - (float(high) + float(low))) / span


def apply_pd_ee_scaling(
    action: np.ndarray, pos_lower: float, pos_upper: float, rot_lower: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce `PDEEPoseController._clip_and_scale_action` in numpy."""
    pos = np.clip(action[:3], -1.0, 1.0)
    pos = 0.5 * (pos_upper + pos_lower) + 0.5 * (pos_upper - pos_lower) * pos
    rot = np.array(action[3:6], dtype=np.float64, copy=True)
    norm = np.linalg.norm(rot)
    if norm > 1.0:
        rot = rot / norm
    rot = rot * float(rot_lower)
    return pos, rot


def normalized_ee_delta(
    delta_pos_m: np.ndarray,
    delta_euler_xyz: np.ndarray,
    gripper: float,
    *,
    pos_lower: float,
    pos_upper: float,
    rot_lower: float,
) -> np.ndarray:
    pos_action = np.clip(unscale_interval(delta_pos_m, pos_lower, pos_upper), -1.0, 1.0)
    rot_action = np.asarray(delta_euler_xyz, dtype=np.float64) / float(rot_lower)
    norm = np.linalg.norm(rot_action)
    if norm > 1.0:
        rot_action = rot_action / norm
    action = np.concatenate([pos_action, rot_action, [gripper]]).astype(np.float32)
    return action


def action_toward_tcp(
    tcp_p: np.ndarray,
    tcp_q: np.ndarray,
    target_p: np.ndarray,
    target_q: np.ndarray,
    base_p: np.ndarray,
    base_q: np.ndarray,
    gripper: float,
    *,
    pos_lower: float,
    pos_upper: float,
    rot_lower: float,
) -> np.ndarray:
    """Closed-loop delta in the default frame `root_translation:root_aligned_body_rotation`."""
    tcp_pb, tcp_qb = world_to_base(tcp_p, tcp_q, base_p, base_q)
    tgt_pb, tgt_qb = world_to_base(target_p, target_q, base_p, base_q)
    delta_p = tgt_pb - tcp_pb
    # Controller composes q_new = q_delta * q_old, so q_delta = q_new * inv(q_old).
    q_delta = quat_mul(quat4(tgt_qb), quat_inv(quat4(tcp_qb)))
    euler = quat_to_euler_xyz(q_delta)
    return normalized_ee_delta(
        delta_p,
        euler,
        gripper,
        pos_lower=pos_lower,
        pos_upper=pos_upper,
        rot_lower=rot_lower,
    )


def hold_action(gripper: float) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[6] = np.float32(gripper)
    return action


def pose_error(p: np.ndarray, q: np.ndarray, target_p: np.ndarray, target_q: np.ndarray) -> tuple[float, float]:
    pos = float(np.linalg.norm(np.asarray(target_p) - np.asarray(p)))
    q_delta = quat_mul(quat4(target_q), quat_inv(quat4(q)))
    rot = float(np.linalg.norm(quat_to_euler_xyz(q_delta)))
    return pos, rot
