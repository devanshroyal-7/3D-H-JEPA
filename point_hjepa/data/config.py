"""YAML collection config."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class CollectConfig:
    env_id: str = "PickCube-v1"
    obs_mode: str = "pointcloud"
    control_mode: str = "pd_ee_delta_pose"
    sim_backend: str = "auto"  # auto | cpu | gpu
    num_envs: int = 1
    num_workers: int = 1
    num_episodes: int = 20
    max_steps: int | None = None
    seed: int = 0
    planner: str = "srmp"
    source_weights: dict = field(
        default_factory=lambda: {"expert": 0.5, "noisy": 0.3, "explore": 0.2}
    )
    noise_epsilon: float = 0.3
    randomize_camera: bool = False
    camera_pos_noise: float = 0.02
    num_points: int = 2048
    store_rgb: bool = False
    downsample: str = "fps"
    voxel_size: float = 0.01
    table_keep: int = 256
    workspace_min: list = field(default_factory=lambda: [-0.2, -0.7, -0.05])
    workspace_max: list = field(default_factory=lambda: [1.0, 0.7, 0.8])
    out: str = "outputs/pickcube.zarr"
    format: str = "zarr"
    context_frames: int = 3
    frameskip: int = 5
    horizon: int = 4
    level2_stride: int = 10
    level2_horizon: int = 2
    probe_srmp_issues: bool = True
    # None: 0.01 m, or 0.03 m on large-mesh tasks such as PickClutterYCB.
    mesh_sampling_resolution: float | None = None
    verbose: bool = False


def load_config(path: str | Path, overrides: dict | None = None) -> CollectConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must be a mapping")
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                raw[key] = value
    known = {item.name for item in fields(CollectConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    return CollectConfig(**raw)
