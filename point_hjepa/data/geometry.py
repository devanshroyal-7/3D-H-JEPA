"""Quaternion and pose helpers. Quaternions are scalar-first (w, x, y, z)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def as_np(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def vec3(value) -> np.ndarray:
    return as_np(value).astype(np.float64).reshape(-1, 3)[0].copy()


def quat4(value) -> np.ndarray:
    q = as_np(value).astype(np.float64).reshape(-1, 4)[0].copy()
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat4(q)
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def quat_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat4(q)
    return Rotation.from_quat([x, y, z, w]).as_euler("XYZ")


def euler_xyz_to_quat(euler: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_euler("XYZ", euler).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def world_to_base(
    p_world: np.ndarray, q_world: np.ndarray, base_p: np.ndarray, base_q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Express a world pose in the robot base frame."""
    rot = quat_to_mat(base_q)
    p_base = rot.T @ (np.asarray(p_world, dtype=np.float64) - np.asarray(base_p, dtype=np.float64))
    q_base = quat_mul(quat_inv(quat4(base_q)), quat4(q_world))
    return p_base, quat4(q_base)


def base_to_world(
    p_base: np.ndarray, q_base: np.ndarray, base_p: np.ndarray, base_q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rot = quat_to_mat(base_q)
    p_world = rot @ np.asarray(p_base, dtype=np.float64) + np.asarray(base_p, dtype=np.float64)
    q_world = quat_mul(quat4(base_q), quat4(q_base))
    return p_world, quat4(q_world)


def apply_local_offset(
    p: np.ndarray, q: np.ndarray, offset_local: np.ndarray
) -> np.ndarray:
    """Translate `p` by a vector expressed in the pose's own frame."""
    return np.asarray(p, dtype=np.float64) + quat_to_mat(q) @ np.asarray(offset_local, dtype=np.float64)


def hand_pose_from_tcp(
    tcp_p: np.ndarray, tcp_q: np.ndarray, tcp_offset_in_hand: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Panda TCP and hand share an orientation. `tcp_offset_in_hand` is TCP minus hand, in the hand frame."""
    rot = quat_to_mat(tcp_q)
    hand_p = np.asarray(tcp_p, dtype=np.float64) - rot @ np.asarray(tcp_offset_in_hand, dtype=np.float64)
    return hand_p, quat4(tcp_q)


def tcp_pose_from_hand(
    hand_p: np.ndarray, hand_q: np.ndarray, tcp_offset_in_hand: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rot = quat_to_mat(hand_q)
    tcp_p = np.asarray(hand_p, dtype=np.float64) + rot @ np.asarray(tcp_offset_in_hand, dtype=np.float64)
    return tcp_p, quat4(hand_q)
