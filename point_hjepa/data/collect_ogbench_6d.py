"""OGBench 6D RGB-XYZ point-cloud collector and MuJoCo depth back-projector.

Supports both:
- ``unsegmented=True`` (default for Point-H-JEPA): keeps table, floor, static
  fixtures, robot arm, and manipulated objects while excluding ONLY invisible
  goal/target mocap markers (and invalid sky depth) so the world model cannot
  cheat while its hierarchy filters background clutter itself.
- ``unsegmented=False`` (JEPA4R-compatible segmented clouds): removes world-body
  background geoms and uses object-stratified sampling down to ``num_points``.

Two collection modes are provided:
- ``wm_collect``: rollouts with OGBench Markov oracles + exploration noise/random
  mixtures (or JEPA4R task policies when available) saved to Zarr v3.
- ``replay_official``: replays official OGBench ``-play-v0`` / ``-noisy-v0``
  trajectories through MuJoCo to lift eye-to-hand RGB-D into Zarr v3.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from point_hjepa.data.zarr_writer import ZarrEpisodeWriter

log = logging.getLogger(__name__)

OGBENCH_TASKS: dict[str, dict[str, Any]] = {
    "cube_single": {
        "env_id": "cube-single-play-v0",
        "task_env_id": "cube-single-v0",
        "display": "Cube-Single",
        "action_dim": 5,
        "episode_len": 200,
        "eval_horizon": 200,
        "num_cubes": 1,
        "policy_type": "gciql",
        "camera": "front_pixels",
        "crop_box": ((0.10, -0.50, -0.02), (0.75, 0.50, 0.60)),
        "affine_center": (0.425, 0.0, 0.2),
        "affine_scale": 4.0,
    },
    "cube_double": {
        "env_id": "cube-double-play-v0",
        "task_env_id": "cube-double-v0",
        "display": "Cube-Double",
        "action_dim": 5,
        "episode_len": 500,
        "eval_horizon": 500,
        "num_cubes": 2,
        "policy_type": "gciql",
        "camera": "front_pixels",
        "crop_box": ((0.10, -0.50, -0.02), (0.75, 0.50, 0.60)),
        "affine_center": (0.425, 0.0, 0.2),
        "affine_scale": 4.0,
    },
    "cube_triple": {
        "env_id": "cube-triple-play-v0",
        "task_env_id": "cube-triple-v0",
        "display": "Cube-Triple",
        "action_dim": 5,
        "episode_len": 1000,
        "eval_horizon": 1000,
        "num_cubes": 3,
        "policy_type": "gciql",
        "camera": "front_pixels",
        "crop_box": ((-0.25, -0.50, -0.02), (0.75, 0.50, 0.90)),
        "affine_center": (0.425, 0.0, 0.2),
        "affine_scale": 4.0,
    },
    "cube_quadruple": {
        "env_id": "cube-quadruple-play-v0",
        "task_env_id": "cube-quadruple-v0",
        "display": "Cube-Quadruple",
        "action_dim": 5,
        "episode_len": 600,
        "eval_horizon": 1000,
        "num_cubes": 4,
        "policy_type": "gciql",
        "camera": "front_pixels",
        "crop_box": ((-0.25, -0.50, -0.02), (0.75, 0.50, 0.90)),
        "affine_center": (0.425, 0.0, 0.2),
        "affine_scale": 4.0,
    },
    "scene": {
        "env_id": "scene-play-v0",
        "task_env_id": "scene-v0",
        "display": "Scene",
        "action_dim": 5,
        "episode_len": 200,
        "eval_horizon": 750,
        "num_cubes": 1,
        "policy_type": "bc",
        "camera": "front_pixels",
        "crop_box": ((0.10, -0.55, -0.02), (0.75, 0.50, 0.60)),
        "affine_center": (0.425, 0.0, 0.2),
        "affine_scale": 4.0,
    },
    "puzzle_3x3": {
        "env_id": "puzzle-3x3-play-v0",
        "task_env_id": "puzzle-3x3-v0",
        "display": "Puzzle 3x3",
        "action_dim": 5,
        "episode_len": 1001,
        "eval_horizon": 1001,
        "num_cubes": 0,
        "policy_type": "dqc",
        "camera": "front_pixels",
        "crop_box": ((0.10, -0.50, -0.02), (0.75, 0.50, 0.60)),
        "affine_center": (0.425, 0.0, 0.2),
        "affine_scale": 4.0,
    },
    "puzzle_4x4": {
        "env_id": "puzzle-4x4-play-v0",
        "task_env_id": "puzzle-4x4-v0",
        "display": "Puzzle 4x4",
        "action_dim": 5,
        "episode_len": 1001,
        "eval_horizon": 1001,
        "num_cubes": 0,
        "policy_type": "dqc",
        "camera": "front_pixels",
        "crop_box": ((0.10, -0.50, -0.02), (0.75, 0.50, 0.60)),
        "affine_center": (0.425, 0.0, 0.2),
        "affine_scale": 4.0,
    },
    "puzzle_4x5": {
        "env_id": "puzzle-4x5-play-v0",
        "task_env_id": "puzzle-4x5-v0",
        "display": "Puzzle 4x5",
        "action_dim": 5,
        "episode_len": 1001,
        "eval_horizon": 1001,
        "num_cubes": 0,
        "policy_type": "dqc",
        "camera": "front_pixels",
        "crop_box": ((0.10, -0.50, -0.02), (0.75, 0.50, 0.60)),
        "affine_center": (0.425, 0.0, 0.2),
        "affine_scale": 4.0,
    },
}


def normalize_ogbench_task(task: str) -> str:
    """Normalize OGBench task aliases (e.g. ``cube-single`` -> ``cube_single``)."""
    key = task.strip().lower().replace("-", "_")
    if key.endswith("_play_v0"):
        key = key[: -len("_play_v0")]
    elif key.endswith("_noisy_v0"):
        key = key[: -len("_noisy_v0")]
    elif key.endswith("_v0"):
        key = key[: -len("_v0")]
    if key not in OGBENCH_TASKS:
        raise ValueError(f"Unknown OGBench task {task!r}. Expected one of {sorted(OGBENCH_TASKS)}")
    return key


def backproject_depth_to_world(
    depth: np.ndarray,
    rgb: np.ndarray,
    fovy_deg: float,
    cam_xpos: np.ndarray,
    cam_xmat: np.ndarray,
    *,
    geom_ids: np.ndarray | None = None,
    drop_geom_ids: set[int] | Sequence[int] | None = None,
    background_geom_ids: set[int] | Sequence[int] | None = None,
    object_geom_ids: set[int] | Sequence[int] | None = None,
    unsegmented: bool = True,
    crop_box: Sequence[Sequence[float]] | None = None,
    voxel_size: float | None = 0.005,
    num_points: int = 4096,
    object_fraction: float = 0.25,
    sort_by_voxel: bool = True,
    max_depth: float = 10.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a MuJoCo RGB-D frame into a world-frame 6D point cloud.

    Matches JEPA4R's MuJoCo camera geometry (``p_cam = [xn*z, yn*z, -z]``,
    ``p_world = p_cam @ R.T + t``). In ``unsegmented=True`` mode, keeps physical
    background/tabletop geoms and samples uniformly, while strictly dropping
    invalid sky hits and invisible mocap goal markers (``drop_geom_ids``).
    """
    depth_arr = np.asarray(depth, dtype=np.float64)
    rgb_arr = np.asarray(rgb, dtype=np.uint8)
    if depth_arr.ndim != 2:
        raise ValueError(f"Expected 2D depth image, got shape {depth_arr.shape}")
    H, W = depth_arr.shape
    if rgb_arr.shape[:2] != (H, W) or rgb_arr.shape[2] < 3:
        raise ValueError(f"RGB shape {rgb_arr.shape} incompatible with depth shape {depth_arr.shape}")

    rng = np.random.default_rng() if rng is None else rng
    z_all = depth_arr.reshape(-1)
    valid = np.isfinite(z_all) & (z_all > 0.0) & (z_all < float(max_depth))

    g_flat: np.ndarray | None = None
    if geom_ids is not None:
        g_flat = np.asarray(geom_ids, dtype=np.int64).reshape(-1)
        if g_flat.shape != z_all.shape:
            raise ValueError(f"geom_ids size {g_flat.shape} does not match depth pixels {z_all.shape}")
        valid &= g_flat >= 0
        if drop_geom_ids:
            drop_arr = np.fromiter((int(x) for x in drop_geom_ids), dtype=np.int64)
            valid &= ~np.isin(g_flat, drop_arr)
        if not unsegmented and background_geom_ids:
            bg_arr = np.fromiter((int(x) for x in background_geom_ids), dtype=np.int64)
            valid &= ~np.isin(g_flat, bg_arr)

    idx = np.flatnonzero(valid)
    n_out = int(num_points)
    if idx.size == 0:
        return np.zeros((n_out, 3), dtype=np.float32), np.zeros((n_out, 3), dtype=np.uint8)

    f = 0.5 * float(H) / np.tan(np.deg2rad(float(fovy_deg)) / 2.0)
    v_grid, u_grid = np.mgrid[0:H, 0:W]
    xn = ((u_grid + 0.5 - W / 2.0) / f).reshape(-1)[idx]
    yn = (-(v_grid + 0.5 - H / 2.0) / f).reshape(-1)[idx]
    z = z_all[idx]

    R = np.asarray(cam_xmat, dtype=np.float64).reshape(3, 3)
    t = np.asarray(cam_xpos, dtype=np.float64).reshape(3)
    p_cam = np.stack([xn * z, yn * z, -z], axis=1)
    pts = p_cam @ R.T + t
    cols = rgb_arr[..., :3].reshape(-1, 3)[idx]

    if g_flat is not None and object_geom_ids:
        obj_arr = np.fromiter((int(x) for x in object_geom_ids), dtype=np.int64)
        is_obj = np.isin(g_flat[idx], obj_arr)
    else:
        is_obj = np.zeros(len(pts), dtype=bool)

    if crop_box is not None:
        lo = np.asarray(crop_box[0], dtype=np.float64)
        hi = np.asarray(crop_box[1], dtype=np.float64)
        mask = np.all((pts >= lo) & (pts <= hi), axis=1)
        pts, cols, is_obj = pts[mask], cols[mask], is_obj[mask]

    if voxel_size is not None and float(voxel_size) > 0.0 and len(pts) > 1:
        q = np.floor(pts / float(voxel_size)).astype(np.int64)
        _, keep = np.unique(q, axis=0, return_index=True)
        pts, cols, is_obj = pts[keep], cols[keep], is_obj[keep]

    if len(pts) == 0:
        return np.zeros((n_out, 3), dtype=np.float32), np.zeros((n_out, 3), dtype=np.uint8)

    if (not unsegmented) and object_geom_ids and np.any(is_obj):
        obj_idx = np.flatnonzero(is_obj)
        other_idx = np.flatnonzero(~is_obj)
        n_obj = min(int(round(float(object_fraction) * n_out)), obj_idx.size)
        if other_idx.size == 0:
            sel = rng.choice(obj_idx, n_out, replace=obj_idx.size < n_out)
        else:
            n_other = n_out - n_obj
            sel = np.concatenate(
                [
                    rng.choice(obj_idx, n_obj, replace=False),
                    rng.choice(other_idx, n_other, replace=other_idx.size < n_other),
                ]
            )
    else:
        sel = rng.choice(len(pts), n_out, replace=len(pts) < n_out)

    pts_sel = pts[sel].astype(np.float32, copy=False)
    cols_sel = cols[sel].astype(np.uint8, copy=False)

    if sort_by_voxel and len(pts_sel) > 1:
        vs = float(voxel_size) if (voxel_size is not None and float(voxel_size) > 0.0) else 0.005
        q_sel = np.floor(pts_sel / vs).astype(np.int64)
        order = np.lexsort((q_sel[:, 2], q_sel[:, 1], q_sel[:, 0]))
        pts_sel = pts_sel[order]
        cols_sel = cols_sel[order]
    else:
        perm = rng.permutation(len(pts_sel))
        pts_sel = pts_sel[perm]
        cols_sel = cols_sel[perm]

    return np.ascontiguousarray(pts_sel, dtype=np.float32), np.ascontiguousarray(cols_sel, dtype=np.uint8)


class OGBenchPointCloudExtractor:
    """Single-camera MuJoCo RGB-D -> 6D point cloud extractor for OGBench envs."""

    def __init__(
        self,
        camera: str = "front_pixels",
        height: int = 224,
        width: int = 224,
        *,
        unsegmented: bool = True,
        num_points: int = 4096,
        voxel_size: float | None = 0.005,
        crop_box: Sequence[Sequence[float]] | None = None,
        object_fraction: float = 0.25,
        sort_by_voxel: bool = True,
    ) -> None:
        self.camera = str(camera)
        self.height = int(height)
        self.width = int(width)
        self.unsegmented = bool(unsegmented)
        self.num_points = int(num_points)
        self.voxel_size = voxel_size
        self.crop_box = crop_box
        self.object_fraction = float(object_fraction)
        self.sort_by_voxel = bool(sort_by_voxel)

        self._model_ref: Any = None
        self._cid: int = -1
        self._fovy: float = 45.0
        self._rflat: Any = None
        self._r2d: Any = None
        self._drop_geoms: set[int] = set()
        self._bg_geoms: set[int] = set()
        self._obj_geoms: set[int] = set()

    def _setup(self, model: Any) -> None:
        import mujoco

        self.close()
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera)
        if cid < 0:
            raise ValueError(f"Camera {self.camera!r} not found in MuJoCo model")
        self._cid = int(cid)
        self._fovy = float(model.cam_fovy[cid])

        self._r2d = mujoco.Renderer(model, height=self.height, width=self.width)
        saved_offsamples = int(model.vis.quality.offsamples)
        model.vis.quality.offsamples = 0
        try:
            self._rflat = mujoco.Renderer(model, height=self.height, width=self.width)
        finally:
            model.vis.quality.offsamples = saved_offsamples

        drop_geoms: set[int] = set()
        bg_geoms: set[int] = set()
        obj_geoms: set[int] = set()
        for g in range(int(model.ngeom)):
            body = int(model.geom_bodyid[g])
            root = int(model.body_rootid[body])
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root) or ""
            alpha = float(model.geom_rgba[g, 3])
            # Strictly exclude invisible mocap targets / hidden goal markers even in unsegmented mode.
            if int(model.body_mocapid[root]) >= 0 or "target" in gname or "target" in bname or alpha <= 0.01:
                drop_geoms.add(g)
                continue
            if int(model.body_weldid[body]) == 0:
                bg_geoms.add(g)
            if (
                "object" in bname
                or "cube" in gname
                or bname.startswith("buttonbox_")
                or bname in ("drawer_base", "window")
            ):
                obj_geoms.add(g)

        self._drop_geoms = drop_geoms
        self._bg_geoms = bg_geoms
        self._obj_geoms = obj_geoms
        self._model_ref = model

    def render_rgb_224(self, model: Any, data: Any) -> np.ndarray:
        """Render anti-aliased 224x224 uint8 RGB image from the depth camera."""
        if self._model_ref is not model or self._r2d is None:
            self._setup(model)
        self._r2d.update_scene(data, camera=self.camera)
        return np.ascontiguousarray(self._r2d.render().copy(), dtype=np.uint8)

    def render_frame(
        self,
        model: Any,
        data: Any,
        *,
        rng: np.random.Generator | None = None,
        return_rgb_224: bool = False,
    ) -> dict[str, np.ndarray]:
        """Render one MuJoCo state into 6D point cloud ``xyz`` / ``rgb`` (and optional ``rgb_224``)."""
        import mujoco

        if self._model_ref is not model or self._rflat is None:
            self._setup(model)

        r = self._rflat
        r.update_scene(data, camera=self.camera)
        rgb_flat = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(data, camera=self.camera)
        depth = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(data, camera=self.camera)
        seg2 = r.render().copy()
        r.disable_segmentation_rendering()

        geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
        seg = np.where(seg2[..., 1] == geom_type, seg2[..., 0], -1)
        R = data.cam_xmat[self._cid].reshape(3, 3).copy()
        t = data.cam_xpos[self._cid].copy()

        xyz, rgb = backproject_depth_to_world(
            depth,
            rgb_flat,
            self._fovy,
            t,
            R,
            geom_ids=seg,
            drop_geom_ids=self._drop_geoms,
            background_geom_ids=self._bg_geoms,
            object_geom_ids=self._obj_geoms,
            unsegmented=self.unsegmented,
            crop_box=self.crop_box,
            voxel_size=self.voxel_size,
            num_points=self.num_points,
            object_fraction=self.object_fraction,
            sort_by_voxel=self.sort_by_voxel,
            rng=rng,
        )
        out: dict[str, np.ndarray] = {"xyz": xyz, "rgb": rgb}
        if return_rgb_224:
            out["rgb_224"] = self.render_rgb_224(model, data)
        return out

    def close(self) -> None:
        if self._r2d is not None:
            self._r2d.close()
            self._r2d = None
        if self._rflat is not None:
            self._rflat.close()
            self._rflat = None
        self._model_ref = None


def extract_ogbench_probe_metadata(env: Any, task: str) -> dict[str, np.ndarray]:
    """Extract ``proprio``, ``ee_point``, and ``achieved_point`` from an OGBench MuJoCo env."""
    import mujoco

    task_key = normalize_ogbench_task(task)
    u = env.unwrapped
    data = u._data
    model = u._model

    if hasattr(u, "compute_ob_info"):
        oi = u.compute_ob_info()
        ee_pos = np.asarray(oi["proprio/effector_pos"], dtype=np.float32).reshape(3)
        yaw = float(np.asarray(oi["proprio/effector_yaw"]).reshape(-1)[0])
        grip = np.asarray(oi["proprio/gripper_opening"], dtype=np.float32).reshape(-1)
    else:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        ee_pos = np.asarray(data.site_xpos[sid] if sid >= 0 else np.zeros(3), dtype=np.float32).reshape(3)
        yaw = 0.0
        grip = np.zeros((1,), dtype=np.float32)

    qpos = np.asarray(data.qpos, dtype=np.float32).reshape(-1)
    qvel = np.asarray(data.qvel, dtype=np.float32).reshape(-1)
    proprio = np.concatenate(
        [ee_pos, np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32), grip, qpos, qvel]
    ).astype(np.float32)

    achieved: np.ndarray | None = None
    if task_key.startswith("cube") or task_key == "scene":
        if hasattr(u, "compute_ob_info"):
            oi = u.compute_ob_info()
            for k in ("privileged/block_0_pos", "privileged/cube_pos"):
                if k in oi:
                    achieved = np.asarray(oi[k], dtype=np.float32).reshape(3)
                    break
        if achieved is None:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint_0")
            if jid >= 0:
                qadr = int(model.jnt_qposadr[jid])
                achieved = np.asarray(data.qpos[qadr : qadr + 3], dtype=np.float32).reshape(3)

    if achieved is None:
        achieved = ee_pos.copy()

    return {
        "proprio": proprio,
        "ee_point": ee_pos,
        "achieved_point": achieved,
    }


def _make_ogbench_env(task_key: str, *, mode: str = "data_collection", max_episode_steps: int | None = None) -> Any:
    import gymnasium
    import importlib

    os.environ.setdefault("MUJOCO_GL", "egl")
    importlib.import_module("ogbench.manipspace")
    spec = OGBENCH_TASKS[task_key]
    steps = int(max_episode_steps or spec["episode_len"])
    return gymnasium.make(
        spec["task_env_id"],
        ob_type="states",
        mode=mode,
        terminate_at_goal=False,
        max_episode_steps=steps,
        width=224,
        height=224,
        visualize_info=False,
    )


def _select_cube_or_scene_oracle(env: Any, task_key: str, ob: np.ndarray, ob_info: dict[str, Any]) -> Any:
    from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle
    from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
    from ogbench.manipspace.oracles.markov.drawer_markov import DrawerMarkovOracle
    from ogbench.manipspace.oracles.markov.window_markov import WindowMarkovOracle

    if task_key.startswith("puzzle"):
        oracle = ButtonMarkovOracle(env=env, min_norm=0.4, gripper_always_closed=True)
    elif task_key == "scene":
        target_task = getattr(env.unwrapped, "_target_task", "cube")
        cls_map = {
            "cube": lambda: CubeMarkovOracle(env=env, min_norm=0.4, max_step=100),
            "button": lambda: ButtonMarkovOracle(env=env, min_norm=0.4),
            "drawer": lambda: DrawerMarkovOracle(env=env, min_norm=0.4),
            "window": lambda: WindowMarkovOracle(env=env, min_norm=0.4),
        }
        oracle = cls_map.get(target_task, cls_map["cube"])()
    else:
        oracle = CubeMarkovOracle(env=env, min_norm=0.4, max_step=100)
    oracle.reset(ob, ob_info)
    return oracle


def collect_ogbench_wm_dataset(cfg: DictConfig) -> Path:
    """Collect exploratory/expert OGBench 6D point-cloud episodes into Zarr v3."""
    task_key = normalize_ogbench_task(str(cfg.task))
    spec = OGBENCH_TASKS[task_key]
    unsegmented = bool(cfg.get("unsegmented", True))
    num_points = int(cfg.get("num_points", 4096 if unsegmented else 1200))
    max_steps = int(cfg.get("episode_len") or spec["episode_len"])
    num_episodes = int(cfg.num_episodes)
    seed = int(cfg.seed)
    crop_box = OmegaConf.to_object(cfg.crop_box) if cfg.get("crop_box") is not None else (
        None if unsegmented else spec["crop_box"]
    )
    voxel_size = float(cfg.voxel_size) if cfg.get("voxel_size") is not None else None
    sort_by_voxel = bool(cfg.get("sort_by_voxel", True))
    xyz_dtype = str(cfg.get("xyz_dtype", "float16"))

    suffix = "unsegmented" if unsegmented else "segmented"
    out_dir = Path(str(cfg.output_dir)) / task_key
    zarr_path = out_dir / f"{task_key}_{suffix}.zarr"

    env = _make_ogbench_env(task_key, mode="data_collection", max_episode_steps=max_steps)
    extractor = OGBenchPointCloudExtractor(
        camera=str(cfg.get("camera", spec["camera"])),
        unsegmented=unsegmented,
        num_points=num_points,
        voxel_size=voxel_size,
        crop_box=crop_box,
        object_fraction=float(cfg.get("object_fraction", 0.25)),
        sort_by_voxel=sort_by_voxel,
    )

    rng = np.random.default_rng(seed)
    writer: ZarrEpisodeWriter | None = None

    try:
        for ep in trange(num_episodes, desc=f"OGBench wm_collect ({task_key}/{suffix})"):
            ep_seed = seed + ep
            np.random.seed(ep_seed)
            ob, ob_info = env.reset(seed=ep_seed)
            u = env.unwrapped
            oracle = _select_cube_or_scene_oracle(env, task_key, ob, ob_info)

            # Match JEPA4R mixture categories: 0=low/noisy expert, 1=high/interrupted, 2=random
            if task_key.startswith("cube"):
                cat = int(rng.choice(4, p=[0.2, 0.4, 0.2, 0.2]))
                noise_std = 0.05 if cat == 0 else (0.2 if cat == 1 else 0.05)
                switch_step = int(rng.integers(max(10, max_steps // 4), max(12, (3 * max_steps) // 4)))
            else:
                cat = int(rng.choice(3, p=[0.5, 0.3, 0.2]))
                noise_std = float(rng.uniform(0.0, 0.2))
                switch_step = int(rng.integers(15, max(16, max_steps - 40)))

            buf_xyz, buf_rgb, buf_act, buf_prop, buf_ach, buf_ee = [], [], [], [], [], []
            for t in range(max_steps):
                frame = extractor.render_frame(u._model, u._data, rng=rng)
                meta = extract_ogbench_probe_metadata(env, task_key)

                use_random = (
                    (task_key.startswith("cube") and cat == 3)
                    or (task_key.startswith("cube") and cat == 2 and t >= switch_step)
                    or (not task_key.startswith("cube") and cat == 2)
                    or (not task_key.startswith("cube") and cat == 1 and t >= switch_step)
                    or (task_key.startswith("puzzle") and rng.random() < 0.2)
                )
                if use_random:
                    action = rng.uniform(-1.0, 1.0, size=(5,)).astype(np.float32)
                else:
                    action = np.asarray(oracle.select_action(ob, ob_info), dtype=np.float32).reshape(5)
                    action = np.clip(action + rng.normal(0.0, noise_std, size=(5,)), -1.0, 1.0).astype(np.float32)

                buf_xyz.append(frame["xyz"])
                buf_rgb.append(frame["rgb"])
                buf_act.append(action)
                buf_prop.append(meta["proprio"])
                buf_ach.append(meta["achieved_point"])
                buf_ee.append(meta["ee_point"])

                ob, _, _, _, ob_info = env.step(action)
                if getattr(oracle, "done", False):
                    next_ob, next_info = u.set_new_target()
                    oracle = _select_cube_or_scene_oracle(env, task_key, next_ob, next_info)

            if writer is None:
                writer = ZarrEpisodeWriter(
                    zarr_path,
                    num_points=num_points,
                    action_dim=5,
                    proprio_dim=int(buf_prop[0].shape[0]),
                    xyz_dtype=xyz_dtype,
                    attrs={
                        "task": task_key,
                        "env_id": spec["env_id"],
                        "mode": "wm_collect",
                        "unsegmented": unsegmented,
                        "num_points": num_points,
                        "camera": str(cfg.get("camera", spec["camera"])),
                        "crop_box": crop_box,
                        "seed": seed,
                    },
                )
            writer.append_episode(
                np.stack(buf_xyz, axis=0),
                np.stack(buf_rgb, axis=0),
                np.stack(buf_act, axis=0),
                np.stack(buf_prop, axis=0),
                np.stack(buf_ach, axis=0),
                np.stack(buf_ee, axis=0),
            )
    finally:
        extractor.close()
        env.close()

    log.info("Collected OGBench wm_collect dataset -> %s (%d episodes)", zarr_path, num_episodes)
    return zarr_path


def collect_ogbench_replay_official(cfg: DictConfig) -> Path:
    """Replay official OGBench ``-play-v0`` trajectories via MuJoCo into 6D Zarr v3."""
    import mujoco
    import ogbench

    task_key = normalize_ogbench_task(str(cfg.task))
    spec = OGBENCH_TASKS[task_key]
    dataset_name = str(cfg.get("dataset_name") or spec["env_id"])
    unsegmented = bool(cfg.get("unsegmented", True))
    num_points = int(cfg.get("num_points", 4096 if unsegmented else 1200))
    crop_box = OmegaConf.to_object(cfg.crop_box) if cfg.get("crop_box") is not None else (
        None if unsegmented else spec["crop_box"]
    )
    voxel_size = float(cfg.voxel_size) if cfg.get("voxel_size") is not None else None
    sort_by_voxel = bool(cfg.get("sort_by_voxel", True))
    xyz_dtype = str(cfg.get("xyz_dtype", "float16"))
    seed = int(cfg.seed)
    max_episodes = cfg.get("num_episodes", None)

    raw = ogbench.load_dataset(dataset_name, compact_dataset=False, add_info=True)
    if "qpos" not in raw or "qvel" not in raw:
        raise KeyError(
            f"Official dataset {dataset_name!r} does not expose 'qpos'/'qvel' needed for MuJoCo replay"
        )

    terminals = np.asarray(raw.get("terminals", raw.get("valids")), dtype=np.float32)
    ep_ends = np.flatnonzero(terminals > 0) + 1
    if ep_ends.size == 0 or ep_ends[-1] != len(raw["actions"]):
        ep_ends = np.append(ep_ends, len(raw["actions"]))
    ep_starts = np.concatenate([np.zeros((1,), dtype=np.int64), ep_ends[:-1]])
    n_eps = len(ep_ends) if max_episodes is None else min(len(ep_ends), int(max_episodes))

    suffix = "unsegmented" if unsegmented else "segmented"
    zarr_path = Path(str(cfg.output_dir)) / task_key / f"{dataset_name}_{suffix}.zarr"
    env = _make_ogbench_env(task_key, mode="data_collection", max_episode_steps=int(spec["episode_len"]))
    extractor = OGBenchPointCloudExtractor(
        camera=str(cfg.get("camera", spec["camera"])),
        unsegmented=unsegmented,
        num_points=num_points,
        voxel_size=voxel_size,
        crop_box=crop_box,
        object_fraction=float(cfg.get("object_fraction", 0.25)),
        sort_by_voxel=sort_by_voxel,
    )
    rng = np.random.default_rng(seed)
    writer: ZarrEpisodeWriter | None = None

    try:
        env.reset(seed=seed)
        u = env.unwrapped
        model, data = u._model, u._data
        for ep in trange(n_eps, desc=f"OGBench replay_official ({dataset_name}/{suffix})"):
            s, e = int(ep_starts[ep]), int(ep_ends[ep])
            buf_xyz, buf_rgb, buf_act, buf_prop, buf_ach, buf_ee = [], [], [], [], [], []
            for idx in range(s, e):
                data.qpos[:] = raw["qpos"][idx]
                data.qvel[:] = raw["qvel"][idx]
                mujoco.mj_forward(model, data)
                frame = extractor.render_frame(model, data, rng=rng)
                meta = extract_ogbench_probe_metadata(env, task_key)
                buf_xyz.append(frame["xyz"])
                buf_rgb.append(frame["rgb"])
                buf_act.append(np.asarray(raw["actions"][idx], dtype=np.float32))
                buf_prop.append(meta["proprio"])
                buf_ach.append(meta["achieved_point"])
                buf_ee.append(meta["ee_point"])

            if writer is None:
                writer = ZarrEpisodeWriter(
                    zarr_path,
                    num_points=num_points,
                    action_dim=int(buf_act[0].shape[0]),
                    proprio_dim=int(buf_prop[0].shape[0]),
                    xyz_dtype=xyz_dtype,
                    attrs={
                        "task": task_key,
                        "env_id": dataset_name,
                        "mode": "replay_official",
                        "unsegmented": unsegmented,
                        "num_points": num_points,
                        "camera": str(cfg.get("camera", spec["camera"])),
                        "crop_box": crop_box,
                        "seed": seed,
                    },
                )
            writer.append_episode(
                np.stack(buf_xyz, axis=0),
                np.stack(buf_rgb, axis=0),
                np.stack(buf_act, axis=0),
                np.stack(buf_prop, axis=0),
                np.stack(buf_ach, axis=0),
                np.stack(buf_ee, axis=0),
            )
    finally:
        extractor.close()
        env.close()

    return zarr_path


@hydra.main(config_path="../../configs", config_name="collect_ogbench", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("OGBench collector config:\n%s", OmegaConf.to_yaml(cfg))
    mode = str(cfg.mode)
    if mode == "wm_collect":
        collect_ogbench_wm_dataset(cfg)
    elif mode == "replay_official":
        collect_ogbench_replay_official(cfg)
    else:
        raise ValueError(f"Unknown OGBench collection mode {mode!r}; expected 'wm_collect' or 'replay_official'")


if __name__ == "__main__":
    main()
