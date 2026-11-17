"""Workarounds for known SRMP integration issues.

Measured on srmp 0.2.2 against the ManiSkill Panda (base translated to
x = -0.615 m, the PickCube mount):

1. Base-pose screw offset
   `plan_screw` after `set_base_pose` does not reach a pose that
   `compute_fk` reports for a nearby configuration. The translation error
   on that check was about 0.18 m (the "about 20 cm" miss). The same screw
   with the planner base left at the identity, and the target expressed in
   the robot base frame, lands within a few millimeters.
   Workaround: never call `set_base_pose` on the planning robot. Transform
   targets and collision objects into the base frame so the planner's world
   frame is the robot base.

2. `read_sim` table
   Importing the PickCube SAPIEN scene makes the reset configuration collide.
   Removing the imported table clears the collision; the table mesh spans
   z in [-0.92, 0] and covers the robot mount, so the top is flush with the
   base. Workaround: drop imported table geometry and add a thin table-top
   slab with a square cutout around the base.

3. `read_sim` ground
   The ground actor sits at the origin while its collision plane is offset
   locally to about z = -0.92. That plane is not used. Workaround: drop
   imported ground and add an explicit slab below the table.

4. Held-object collision near the table
   Collision checks with a grasped cube still overlapping the table are not
   reliable. Workaround: after the fingers close, lift `POST_GRASP_LIFT_M`
   (5 cm) with collision checking off, and only then attach the object and
   plan the transport.
"""

from __future__ import annotations

import numpy as np

# Historical magnitude of the set_base_pose / plan_screw miss. The probe
# records the value measured in this process; this constant is the threshold
# above which the workaround is considered necessary.
KNOWN_SCREW_BASE_ERROR_M = 0.20
POST_GRASP_LIFT_M = 0.05
TABLE_TOP_THICKNESS_M = 0.03
TABLE_TOP_CLEARANCE_M = 0.002
TABLE_BASE_CUTOUT_HALF_M = 0.18
GROUND_SLAB_CENTER_Z = -0.08
GROUND_SLAB_THICKNESS_M = 0.02
TABLE_NAME_TOKENS = ("table",)
GROUND_NAME_TOKENS = ("ground", "floor")


def compensate_screw_target(desired_p: np.ndarray, bias_p: np.ndarray) -> np.ndarray:
    """Subtract a measured plan_screw translation bias from a target position.

    Used only as a residual correction. The primary fix is to avoid
    `set_base_pose` entirely; see the module docstring.
    """
    return np.asarray(desired_p, dtype=np.float64) - np.asarray(bias_p, dtype=np.float64)


def is_table_actor(name: str) -> bool:
    low = name.lower()
    return any(token in low for token in TABLE_NAME_TOKENS)


def is_ground_actor(name: str) -> bool:
    low = name.lower()
    return any(token in low for token in GROUND_NAME_TOKENS)


def table_slabs_with_base_cutout(
    mins: np.ndarray,
    maxs: np.ndarray,
    *,
    cutout_half: float = TABLE_BASE_CUTOUT_HALF_M,
    thickness: float = TABLE_TOP_THICKNESS_M,
    top_clearance: float = TABLE_TOP_CLEARANCE_M,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Thin table-top boxes in the planner frame, with a hole around the base.

    `mins` and `maxs` are the table AABB in the robot base frame. The slab
    uses only the top `thickness` meters of that AABB, lowered by
    `top_clearance` so it does not sit flush with the robot base at z = 0.
    The square cutout is centered on the base origin. Each item is
    `(center_xyz, size_xyz)`.
    """
    mins = np.asarray(mins, dtype=np.float64)
    maxs = np.asarray(maxs, dtype=np.float64)
    top = float(maxs[2]) - float(top_clearance)
    z_center = top - float(thickness) / 2.0
    size_z = float(thickness)
    hole_min = np.array([-cutout_half, -cutout_half], dtype=np.float64)
    hole_max = np.array([cutout_half, cutout_half], dtype=np.float64)
    x0, y0 = float(mins[0]), float(mins[1])
    x1, y1 = float(maxs[0]), float(maxs[1])
    hx0, hy0 = float(hole_min[0]), float(hole_min[1])
    hx1, hy1 = float(hole_max[0]), float(hole_max[1])
    # Four rectangles around the hole, clipped to the table AABB.
    rects = [
        (x0, y0, min(hx0, x1), y1),  # left of the hole
        (max(hx1, x0), y0, x1, y1),  # right (in front, +x) of the hole
        (max(hx0, x0), y0, min(hx1, x1), min(hy0, y1)),  # in front of the hole in -y
        (max(hx0, x0), max(hy1, y0), min(hx1, x1), y1),
    ]
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    for xa, ya, xb, yb in rects:
        if xb - xa < 1e-3 or yb - ya < 1e-3:
            continue
        center = np.array([(xa + xb) / 2.0, (ya + yb) / 2.0, z_center], dtype=np.float64)
        size = np.array([xb - xa, yb - ya, size_z], dtype=np.float64)
        boxes.append((center, size))
    return boxes


def cutout_covers_base(boxes: list[tuple[np.ndarray, np.ndarray]], half: float) -> bool:
    """True when no slab overlaps the open square around the base origin."""
    for center, size in boxes:
        xmin = center[0] - size[0] / 2.0
        xmax = center[0] + size[0] / 2.0
        ymin = center[1] - size[1] / 2.0
        ymax = center[1] + size[1] / 2.0
        overlap_x = min(xmax, half) - max(xmin, -half)
        overlap_y = min(ymax, half) - max(ymin, -half)
        if overlap_x > 1e-6 and overlap_y > 1e-6:
            return False
    return True


def ground_slab(
    mins_xy: np.ndarray,
    maxs_xy: np.ndarray,
    *,
    center_z: float = GROUND_SLAB_CENTER_Z,
    thickness: float = GROUND_SLAB_THICKNESS_M,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.array(
        [
            0.5 * (float(mins_xy[0]) + float(maxs_xy[0])),
            0.5 * (float(mins_xy[1]) + float(maxs_xy[1])),
            center_z,
        ],
        dtype=np.float64,
    )
    size = np.array(
        [
            max(float(maxs_xy[0] - mins_xy[0]), 0.05),
            max(float(maxs_xy[1] - mins_xy[1]), 0.05),
            thickness,
        ],
        dtype=np.float64,
    )
    return center, size
