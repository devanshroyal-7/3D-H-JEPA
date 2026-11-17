"""Environment construction, camera noise, and segmentation grouping."""

from __future__ import annotations

import os

import numpy as np

from point_hjepa.data.planners.workarounds import is_ground_actor, is_table_actor


def ensure_vulkan() -> None:
    """Point clouds are rendered even for CPU physics, so a Vulkan ICD is required.

    On a machine without an NVIDIA device, lavapipe (Mesa's software rasterizer)
    is selected when the user has not already chosen an ICD.
    """
    os.environ.setdefault("SAPIEN_NO_DISPLAY", "1")
    if os.environ.get("VK_ICD_FILENAMES"):
        return
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.cuda.is_available():
        return
    lavapipe = "/usr/share/vulkan/icd.d/lvp_icd.json"
    if os.path.exists(lavapipe):
        os.environ["VK_ICD_FILENAMES"] = lavapipe


def resolve_sim_backend(requested: str) -> str:
    import torch

    key = (requested or "auto").lower()
    if key == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"
    if key in ("gpu", "cuda", "physx_cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "sim_backend=gpu was requested but CUDA is not available. "
                "Refusing to silently switch backends."
            )
        return "gpu"
    if key in ("cpu", "physx_cpu"):
        return "cpu"
    raise ValueError(f"Unknown sim_backend {requested!r}")


class SrmpRequiresCpuScene(RuntimeError):
    """SRMP's read path is a CPU SAPIEN scene, not a GPU-batched ManiSkill scene."""


def make_env(cfg, sim_backend: str, num_envs: int):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (register tasks)

    render_backend = "gpu" if sim_backend == "gpu" else "cpu"
    env = gym.make(
        cfg.env_id,
        obs_mode=cfg.obs_mode,
        control_mode=cfg.control_mode,
        sim_backend="physx_cuda" if sim_backend == "gpu" else "physx_cpu",
        render_backend=render_backend,
        num_envs=num_envs,
        max_episode_steps=cfg.max_steps,
        sensor_configs={"shader_pack": "minimal"},
    )
    return env


def randomize_camera(env, rng: np.random.Generator, noise_m: float) -> None:
    import sapien

    camera = env.unwrapped._sensors["base_camera"].camera
    local = camera.get_local_pose()
    position = np.asarray(local.p, dtype=np.float64) + rng.normal(0.0, noise_m, size=3)
    camera.set_local_pose(sapien.Pose(p=position, q=np.asarray(local.q, dtype=np.float64)))


def segmentation_groups(env, actor_attrs: tuple[str, ...]) -> tuple[set[int], set[int], set[int]]:
    from mani_skill.utils.structs import Link

    names: set[str] = set()
    for attr in actor_attrs:
        actor = getattr(env.unwrapped, attr, None)
        if actor is not None and getattr(actor, "name", None):
            names.add(str(actor.name))
    robot: set[int] = set()
    objects: set[int] = set()
    table: set[int] = set()
    for seg_id, obj in env.unwrapped.segmentation_id_map.items():
        name = str(getattr(obj, "name", ""))
        if isinstance(obj, Link) or "panda" in name:
            robot.add(int(seg_id))
        elif is_table_actor(name):
            table.add(int(seg_id))
        elif is_ground_actor(name) or "goal" in name:
            continue
        elif name in names or any(token and token in name for token in names):
            objects.add(int(seg_id))
    return robot, objects, table
