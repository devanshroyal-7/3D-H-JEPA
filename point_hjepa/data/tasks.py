"""Task tiers for the point-cloud collection pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from point_hjepa.data.geometry import vec3


class TaskUnavailable(RuntimeError):
    """Raised when a configured task cannot be collected in this environment."""


# srmp's VAMP default surface sample spacing. Coarser (larger) values use
# fewer spheres for mesh obstacles and attached meshes.
FINE_MESH_SAMPLING_RESOLUTION_M = 0.01
COARSE_MESH_SAMPLING_RESOLUTION_M = 0.03


@dataclass(frozen=True)
class TaskSpec:
    env_id: str
    tier: int
    max_steps: int
    expert: str
    actor_attrs: tuple[str, ...]
    feasible: bool = True
    reason: str = ""
    # YCB / clutter meshes are large; VAMP's sphere samples grow with area.
    large_meshes: bool = False


_SETTABLE_REASON = (
    "ManiSkill-HAB SetTable is not registered in mani_skill 3.0.1 (gym.spec "
    "'SetTable-v1' is missing). The task lives in the separate ManiSkill-HAB "
    "package and needs ReplicaCAD rearranged scenes plus a mobile manipulator. "
    "The SRMP Panda tabletop planner does not cover that embodiment, so tier-3 "
    "SetTable is stubbed rather than silently replaced with another planner."
)

_CLUTTER_ASSET_REASON = (
    "PickClutterYCB-v1 is registered, but its episode config and YCB meshes are "
    "not in this environment. Download them with "
    "`python -m mani_skill.utils.download_asset ycb` and "
    "`python -m mani_skill.utils.download_asset pick_clutter_ycb_configs`. "
    "ManiSkill 3.0.1 also implements PickClutterYCBEnv.evaluate as constant "
    "False and leaves obs['extra'] empty in pointcloud mode, so success labels "
    "and goal_coords cannot come from the observation until that task is extended. "
    "Tier 3 was not run."
)


TASKS: dict[str, TaskSpec] = {
    "PickCube-v1": TaskSpec("PickCube-v1", 1, 50, "pick", ("cube",)),
    "PushCube-v1": TaskSpec("PushCube-v1", 1, 50, "push", ("obj",)),
    "StackCube-v1": TaskSpec("StackCube-v1", 2, 150, "stack", ("cubeA", "cubeB")),
    "PullCubeTool-v1": TaskSpec("PullCubeTool-v1", 2, 200, "pull", ("cube", "l_shape_tool")),
    "PegInsertionSide-v1": TaskSpec("PegInsertionSide-v1", 2, 200, "peg", ("peg",)),
    "PickClutterYCB-v1": TaskSpec(
        "PickClutterYCB-v1",
        3,
        250,
        "clutter",
        ("target_object",),
        feasible=True,
        reason=_CLUTTER_ASSET_REASON,
        large_meshes=True,
    ),
    "SetTable-v1": TaskSpec(
        "SetTable-v1",
        3,
        300,
        "settable",
        (),
        feasible=False,
        reason=_SETTABLE_REASON,
    ),
}


def resolve_mesh_sampling_resolution(env_id: str, override: float | None) -> float:
    """Spacing in metres for VAMP mesh samples.

    An explicit config value wins. Otherwise large-mesh tasks (YCB clutter)
    use a coarser spacing than srmp's 1 cm default, which keeps the sphere
    count, and VAMP's memory, down.
    """
    if override is not None:
        return float(override)
    spec = TASKS.get(env_id)
    if spec is not None and spec.large_meshes:
        return COARSE_MESH_SAMPLING_RESOLUTION_M
    return FINE_MESH_SAMPLING_RESOLUTION_M


def clutter_assets_present() -> bool:
    try:
        import mani_skill
    except ImportError:
        return False
    root = mani_skill.ASSET_DIR
    config = os.path.join(root, "tasks", "pick_clutter", "ycb_train_5k.json.gz")
    meshes = os.path.join(root, "assets", "mani_skill2_ycb")
    return os.path.exists(config) and os.path.exists(meshes)


def require_task(env_id: str) -> TaskSpec:
    if env_id not in TASKS:
        known = ", ".join(sorted(TASKS))
        raise TaskUnavailable(f"No collector spec for {env_id}. Known tasks: {known}")
    spec = TASKS[env_id]
    if not spec.feasible:
        raise TaskUnavailable(spec.reason)
    if env_id == "PickClutterYCB-v1" and not clutter_assets_present():
        raise TaskUnavailable(spec.reason)
    return spec


def _z_half(value) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size >= 3:
        return float(array[2])
    return float(array[0])


def fallback_goal(env, env_id: str) -> np.ndarray:
    """Goal position when pointcloud mode omits it from obs['extra'].

    PickCube always puts ``goal_pos`` in extra. Push, stack, peg, and pull only
    add goal fields when ``"state"`` is in the obs mode, which ``pointcloud``
    is not. Those tasks record the simulator goal and tag the source.
    """
    unwrapped = env.unwrapped
    if env_id == "PickCube-v1":
        return vec3(unwrapped.goal_site.pose.p).astype(np.float32)
    if env_id == "PushCube-v1":
        return vec3(unwrapped.goal_region.pose.p).astype(np.float32)
    if env_id == "StackCube-v1":
        bottom = vec3(unwrapped.cubeB.pose.p)
        height = 2.0 * _z_half(unwrapped.cube_half_size)
        return np.array([bottom[0], bottom[1], bottom[2] + height], dtype=np.float32)
    if env_id == "PegInsertionSide-v1":
        pose = unwrapped.box_hole_pose if hasattr(unwrapped, "box_hole_pose") else unwrapped.goal_pose
        return vec3(pose.p).astype(np.float32)
    if env_id == "PullCubeTool-v1":
        base = vec3(unwrapped.agent.robot.pose.p)
        cube = vec3(unwrapped.cube.pose.p)
        return np.array([base[0] + 0.35, cube[1], cube[2]], dtype=np.float32)
    if env_id == "PickClutterYCB-v1":
        return vec3(unwrapped.goal_pos).astype(np.float32)
    return np.zeros(3, dtype=np.float32)
