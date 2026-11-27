"""End-to-end OGBench feature caching and downstream evaluation harness."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from point_hjepa.data.collect_ogbench_6d import OGBENCH_TASKS, OGBenchPointCloudExtractor, normalize_ogbench_task
from point_hjepa.eval.encoders import FrozenObsEncoder, build_encoder
from point_hjepa.eval.jepa4r_adapter import build_jepa4r_native_command
from point_hjepa.eval.ogbench_policies import (
    FLAREBCConfig,
    GCIQLConfig,
    GCIQLGoalSampler,
    build_flare_features,
    evaluate_gciql_online,
    evaluate_scene_online,
    train_flare_bc,
    train_gciql,
)


def _encoder_cache_tag(
    method: str,
    *,
    latent_mode: str = "concat",
    unsegmented: bool = True,
    num_points: int = 4096,
) -> str:
    if method == "point_hjepa":
        seg = "unseg" if unsegmented else "seg"
        return f"point_hjepa_{latent_mode}_{seg}_{num_points}"
    return method


def cache_ogbench_features(
    task: str,
    encoder: FrozenObsEncoder,
    cache_dir: str | Path,
    *,
    unsegmented: bool = True,
    num_points: int = 4096,
    latent_mode: str = "concat",
    splits: tuple[str, ...] = ("train", "val"),
    max_episodes: int | None = None,
) -> Path:
    """Replay official OGBench policy trajectories and cache frozen latents per episode."""
    import gymnasium
    import importlib
    import mujoco
    import ogbench

    os.environ.setdefault("MUJOCO_GL", "egl")
    importlib.import_module("ogbench.manipspace")

    task_key = normalize_ogbench_task(task)
    spec = OGBENCH_TASKS[task_key]
    tag = _encoder_cache_tag(encoder.name if encoder.name != "state" else "state", latent_mode=latent_mode, unsegmented=unsegmented, num_points=num_points)
    out_root = Path(cache_dir) / task_key / tag
    out_root.mkdir(parents=True, exist_ok=True)

    if encoder.modality == "state":
        manifest = {"task": task_key, "encoder": encoder.name, "modality": "state", "latent_dim": encoder.latent_dim}
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return out_root

    extractor = OGBenchPointCloudExtractor(
        camera=spec["camera"],
        unsegmented=unsegmented,
        num_points=num_points,
        voxel_size=0.005,
        crop_box=None if unsegmented else spec["crop_box"],
        sort_by_voxel=True,
    )
    env = gymnasium.make(spec["task_env_id"], ob_type="states", mode="data_collection", width=224, height=224)

    try:
        env.reset(seed=0)
        u = env.unwrapped
        model, data = u._model, u._data

        for split in splits:
            ds_name = spec["env_id"] if split == "train" else f"{spec['env_id']}-val"
            raw = ogbench.load_dataset(ds_name, compact_dataset=False, add_info=True)
            terminals = np.asarray(raw.get("terminals", raw.get("valids")), dtype=np.float32)
            ep_ends = np.flatnonzero(terminals > 0) + 1
            if ep_ends.size == 0 or ep_ends[-1] != len(raw["actions"]):
                ep_ends = np.append(ep_ends, len(raw["actions"]))
            ep_starts = np.concatenate([np.zeros((1,), dtype=np.int64), ep_ends[:-1]])
            n_eps = len(ep_ends) if max_episodes is None else min(len(ep_ends), int(max_episodes))

            rng = np.random.default_rng(0 if split == "train" else 1)
            for ep in range(n_eps):
                ep_dir = out_root / split / f"episode_{ep:06d}"
                lat_path = ep_dir / "latent.npy"
                if lat_path.is_file():
                    continue
                s, e = int(ep_starts[ep]), int(ep_ends[ep])
                obs_seq = []
                for idx in range(s, e):
                    data.qpos[:] = raw["qpos"][idx]
                    data.qvel[:] = raw["qvel"][idx]
                    mujoco.mj_forward(model, data)
                    frame = extractor.render_frame(model, data, rng=rng, return_rgb_224=(encoder.modality == "rgb"))
                    if encoder.modality == "pcd":
                        obs_seq.append(np.concatenate([frame["xyz"], frame["rgb"].astype(np.float32)], axis=-1))
                    else:
                        obs_seq.append(frame["rgb_224"])
                latents = encoder.encode_batch(np.stack(obs_seq, axis=0))
                ep_dir.mkdir(parents=True, exist_ok=True)
                np.save(lat_path, latents.astype(np.float32))
    finally:
        extractor.close()
        env.close()

    manifest = {
        "task": task_key,
        "encoder": encoder.name,
        "modality": encoder.modality,
        "latent_dim": encoder.latent_dim,
        "unsegmented": unsegmented,
        "num_points": num_points,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_root


def _load_cached_split(cache_root: Path, split: str, n_episodes: int, latent_dim: int) -> np.ndarray:
    chunks = []
    for ep in range(n_episodes):
        arr = np.load(cache_root / split / f"episode_{ep:06d}" / "latent.npy", allow_pickle=False)
        if arr.shape[1] != latent_dim:
            raise ValueError(f"Cached latent dim {arr.shape[1]} != expected {latent_dim}")
        chunks.append(arr)
    return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)


def run_ogbench_evaluation(
    task: str,
    method: str,
    *,
    wm_ckpt: str | Path | None = None,
    latent_mode: str = "concat",
    unsegmented: bool = True,
    num_points: int | None = None,
    baseline_ckpt: str | Path | None = None,
    jepa4r_root: str | Path = "third_party/JEPA4R",
    checkpoints_root: str | Path = "checkpoints/jepa4r",
    cache_dir: str | Path = "outputs/ogbench_cache",
    out_dir: str | Path = "outputs/ogbench_eval",
    backend: str = "torch",
    train_steps: int = 1_000_000,
    bc_epochs: int = 100,
    episodes_per_task: int | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Run full OGBench downstream caching, policy training, and online 5-task evaluation."""
    task_key = normalize_ogbench_task(task)
    spec = OGBENCH_TASKS[task_key]
    pts = int(num_points or (4096 if (method == "point_hjepa" and unsegmented) else 1200))
    run_out = Path(out_dir) / task_key / method
    run_out.mkdir(parents=True, exist_ok=True)

    if backend == "jepa4r_native":
        cmd = build_jepa4r_native_command(task_key, method, jepa4r_root=jepa4r_root)
        subprocess.run(cmd, cwd=str(jepa4r_root), check=True)
        metrics_file = Path(jepa4r_root) / "outputs" / "train_policy" / task_key / method / "point_hjepa_eval" / "metrics.json"
        if metrics_file.is_file():
            res = json.loads(metrics_file.read_text())
            (run_out / "metrics.json").write_text(json.dumps(res, indent=2) + "\n")
            return res
        return {"task": task_key, "method": method, "backend": "jepa4r_native", "status": "completed"}

    if backend != "torch":
        raise ValueError(f"Unknown backend {backend!r}; choose 'torch' or 'jepa4r_native'")

    import gymnasium
    import importlib
    import ogbench

    os.environ.setdefault("MUJOCO_GL", "egl")
    importlib.import_module("ogbench.manipspace")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state_dim = 19 + 9 * spec["num_cubes"] if spec["num_cubes"] > 0 else None
    encoder = build_encoder(
        method,
        task_key,
        wm_ckpt=wm_ckpt,
        latent_mode=latent_mode,
        state_dim=state_dim,
        baseline_ckpt=baseline_ckpt,
        jepa4r_root=jepa4r_root,
        checkpoints_root=checkpoints_root,
        device=dev,
    )

    if spec["policy_type"] == "gciql":
        raw_train = ogbench.load_dataset(spec["env_id"], compact_dataset=True, add_info=False)
        ep_len = int(spec["eval_horizon"]) + 1
        n_eps = len(raw_train["observations"]) // ep_len
        cache_root = cache_ogbench_features(
            task_key,
            encoder,
            cache_dir,
            unsegmented=unsegmented,
            num_points=pts,
            latent_mode=latent_mode,
            splits=("train",),
            max_episodes=n_eps,
        )
        features = None if encoder.modality == "state" else _load_cached_split(cache_root, "train", n_eps, encoder.latent_dim)
        ep_ends = np.arange(1, n_eps + 1, dtype=np.int64) * ep_len
        obs_dim = encoder.latent_dim if encoder.modality == "state" else encoder.latent_dim + 19
        goal_dim = 3 * spec["num_cubes"] if encoder.modality == "state" else encoder.latent_dim + 3 * spec["num_cubes"]
        gcfg = GCIQLConfig(obs_dim=obs_dim, goal_dim=goal_dim, action_dim=5)
        sampler = GCIQLGoalSampler(
            raw_train["observations"],
            raw_train["actions"],
            ep_ends,
            num_cubes=spec["num_cubes"],
            features=features,
            cfg=gcfg,
        )
        policy, evals = train_gciql(sampler, gcfg, train_steps=train_steps, device=dev)
        extractor = OGBenchPointCloudExtractor(
            camera=spec["camera"],
            unsegmented=unsegmented,
            num_points=pts,
            crop_box=None if unsegmented else spec["crop_box"],
        )
        n_eval_eps = int(episodes_per_task or 50)
        try:
            eval_res = evaluate_gciql_online(
                policy,
                lambda: gymnasium.make(spec["task_env_id"], ob_type="states", mode="task", width=224, height=224),
                encoder,
                extractor,
                episodes_per_task=n_eval_eps,
                horizon=int(spec["eval_horizon"]),
                modality=encoder.modality,
                num_cubes=spec["num_cubes"],
                device=dev,
            )
        finally:
            extractor.close()
        result = {
            "task": task_key,
            "method": method,
            "latent_mode": latent_mode if method == "point_hjepa" else None,
            "unsegmented": unsegmented if method == "point_hjepa" else False,
            "evals": evals,
            **eval_res,
        }
    elif spec["policy_type"] == "bc":
        # Scene BC evaluation across 5 official tasks
        from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle
        from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
        from ogbench.manipspace.oracles.markov.drawer_markov import DrawerMarkovOracle
        from ogbench.manipspace.oracles.markov.window_markov import WindowMarkovOracle

        extractor = OGBenchPointCloudExtractor(
            camera=spec["camera"],
            unsegmented=unsegmented,
            num_points=pts,
            crop_box=None if unsegmented else spec["crop_box"],
        )
        n_eval_eps = int(episodes_per_task or 25)
        per_task: dict[str, float] = {}
        try:
            for tid in range(1, 6):
                # Collect 25 official task demos on the fly
                demo_env = gymnasium.make("scene-v0", ob_type="states", mode="task", max_episode_steps=750, width=224, height=224)
                feat_chunks, act_chunks = [], []
                try:
                    for ep in range(25):
                        ob, info = demo_env.reset(seed=1000 * tid + ep, options={"task_id": tid})
                        u = demo_env.unwrapped
                        t_name = getattr(u, "_target_task", "cube")
                        oracle_cls = {"cube": CubeMarkovOracle, "button": ButtonMarkovOracle, "drawer": DrawerMarkovOracle, "window": WindowMarkovOracle}[t_name]
                        oracle = oracle_cls(env=demo_env, min_norm=0.4)
                        oracle.reset(ob, info)
                        z_list, prop_list, act_list = [], [], []
                        for _ in range(750):
                            oi = u.compute_ob_info()
                            yaw = float(np.asarray(oi["proprio/effector_yaw"]).reshape(-1)[0])
                            prop = np.concatenate([
                                np.asarray(oi["proprio/effector_pos"], dtype=np.float32),
                                np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32),
                                np.asarray(oi["proprio/gripper_opening"], dtype=np.float32).reshape(-1),
                            ])[:6]
                            if encoder.modality == "state":
                                z_cur = np.asarray(ob, dtype=np.float32)
                            else:
                                fr = extractor.render_frame(u._model, u._data, return_rgb_224=(encoder.modality == "rgb"))
                                f_obs = np.concatenate([fr["xyz"], fr["rgb"].astype(np.float32)], axis=-1) if encoder.modality == "pcd" else fr["rgb_224"]
                                z_cur = encoder.encode_one(f_obs)
                            act = np.asarray(oracle.select_action(ob, info), dtype=np.float32)
                            z_list.append(z_cur)
                            prop_list.append(prop)
                            act_list.append(act)
                            ob, _, term, trunc, info = demo_env.step(act)
                            if term or trunc:
                                break
                        feat_chunks.append(build_flare_features(np.stack(z_list), np.stack(prop_list), frame_stack=3, proprio_dim=6))
                        act_chunks.append(np.stack(act_list))
                finally:
                    demo_env.close()

                bcfg = FLAREBCConfig(latent_dim=encoder.latent_dim, proprio_dim=6, action_dim=5, epochs=bc_epochs)
                policy, _ = train_flare_bc(np.concatenate(feat_chunks), np.concatenate(act_chunks), bcfg, device=dev)
                per_task[f"scene-task{tid}-v0"] = evaluate_scene_online(
                    policy,
                    lambda task_id=tid: gymnasium.make("scene-v0", ob_type="states", mode="task", max_episode_steps=750, width=224, height=224),
                    encoder,
                    extractor,
                    task_id=tid,
                    episodes=n_eval_eps,
                    modality=encoder.modality,
                    device=dev,
                )
        finally:
            extractor.close()
        result = {
            "task": task_key,
            "method": method,
            "latent_mode": latent_mode if method == "point_hjepa" else None,
            "unsegmented": unsegmented if method == "point_hjepa" else False,
            "success": sum(per_task.values()) / max(1, len(per_task)),
            "success_per_task": per_task,
            "evals": [],
        }
    else:
        raise NotImplementedError(
            f"PyTorch downstream policy for {task_key!r} (DQC) is not ported; use --backend jepa4r_native"
        )

    (run_out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
