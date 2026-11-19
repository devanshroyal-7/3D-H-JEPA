"""Collect point-cloud episodes for 3D Point H-JEPA.

CPU simulation is the SRMP path: the planner reads a SAPIEN scene, so expert
and noisy-expert episodes run one environment per process. When a GPU is
present and every episode is random exploration, collection uses a vectorized
GPU env. ``sim_backend: gpu`` together with expert sources raises
``SrmpRequiresCpuScene`` instead of switching to mplib.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

import numpy as np

from point_hjepa.data.config import CollectConfig, load_config
from point_hjepa.data.control import hold_action
from point_hjepa.data.dataset import PointCloudWindowDataset
from point_hjepa.data.envs import (
    SrmpRequiresCpuScene,
    ensure_vulkan,
    make_env,
    randomize_camera,
    resolve_sim_backend,
    segmentation_groups,
)
from point_hjepa.data.experts import ExpertPolicy
from point_hjepa.data.geometry import quat4, vec3, world_to_base
from point_hjepa.data.planners import create_planner
from point_hjepa.data.preprocess import PreprocessConfig, preprocess_pointclouds
from point_hjepa.data.sources import SOURCE_CODE, assign_sources, source_counts
from point_hjepa.data.tasks import TASKS, fallback_goal, require_task, resolve_mesh_sampling_resolution
from point_hjepa.data.writer import AsyncEpisodeWriter


def _as_bool(value) -> bool:
    return bool(np.asarray(value).reshape(-1)[0])


def _flag(info, key: str) -> bool:
    if not info or key not in info:
        return False
    return _as_bool(info[key])


def _goal_vector(obs, env, env_id: str) -> tuple[np.ndarray, str]:
    extra = obs.get("extra", {}) if isinstance(obs, dict) else {}
    parts = []
    keys = []
    for key, value in extra.items():
        if "goal" not in key.lower():
            continue
        array = np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)
        array = array.astype(np.float32)
        if array.ndim >= 2:
            array = array[0]
        array = array.reshape(-1)
        if array.size:
            parts.append(array)
            keys.append(key)
    if parts:
        return np.concatenate(parts), "obs.extra:" + ",".join(keys)
    return fallback_goal(env, env_id).astype(np.float32), "env"


def _robot_state(obs, env) -> np.ndarray:
    qpos = np.asarray(obs["agent"]["qpos"].detach().cpu().numpy(), dtype=np.float32).reshape(-1)
    qvel = np.asarray(obs["agent"]["qvel"].detach().cpu().numpy(), dtype=np.float32).reshape(-1)
    pose = env.unwrapped.agent.tcp.pose
    tcp = np.concatenate([vec3(pose.p), quat4(pose.q)]).astype(np.float32)
    return np.concatenate([qpos, qvel, tcp])


def _snapshot(obs, env, cfg: CollectConfig, groups, rng) -> dict:
    cloud = obs["pointcloud"]
    pre = preprocess_pointclouds(
        cloud["xyzw"],
        cloud["segmentation"],
        rgb=cloud["rgb"] if cfg.store_rgb else None,
        base_p=vec3(env.unwrapped.agent.robot.pose.p),
        base_q=quat4(env.unwrapped.agent.robot.pose.q),
        robot_ids=groups[0],
        object_ids=groups[1],
        table_ids=groups[2],
        config=PreprocessConfig(
            num_points=cfg.num_points,
            workspace_min=tuple(cfg.workspace_min),
            workspace_max=tuple(cfg.workspace_max),
            table_keep=cfg.table_keep,
            downsample=cfg.downsample,
            voxel_size=cfg.voxel_size,
            store_rgb=cfg.store_rgb,
        ),
        rng=rng,
    )
    goal, goal_source = _goal_vector(obs, env, cfg.env_id)
    row = {
        "points": pre["points"][0],
        "point_count": int(pre["point_count"][0]),
        "goal": goal,
        "goal_source": goal_source,
        "robot_state": _robot_state(obs, env),
    }
    if cfg.store_rgb:
        row["rgb"] = pre["rgb"][0]
    return row


def _explore_action(env, rng: np.random.Generator, cfg: CollectConfig) -> np.ndarray:
    action = rng.uniform(-1.0, 1.0, size=7).astype(np.float32)
    tcp = vec3(env.unwrapped.agent.tcp.pose.p)
    base_p = vec3(env.unwrapped.agent.robot.pose.p)
    base_q = quat4(env.unwrapped.agent.robot.pose.q)
    local, _ = world_to_base(tcp, np.array([1.0, 0.0, 0.0, 0.0]), base_p, base_q)
    low = np.asarray(cfg.workspace_min, dtype=np.float64)
    high = np.asarray(cfg.workspace_max, dtype=np.float64)
    for axis in range(3):
        if local[axis] < low[axis]:
            action[axis] = 1.0
        elif local[axis] > high[axis]:
            action[axis] = -1.0
    if rng.random() > 0.15:
        action[6] = np.float32(1.0 if rng.random() < 0.5 else -1.0)
    return action


def _noisy(action: np.ndarray, rng: np.random.Generator, epsilon: float) -> np.ndarray:
    noisy = action + rng.normal(0.0, epsilon, size=action.shape)
    return np.clip(noisy, -1.0, 1.0).astype(np.float32)


def run_episode(env, planner, cfg: CollectConfig, episode_index: int, source: str, groups) -> dict:
    rng = np.random.default_rng(cfg.seed + 10007 * (episode_index + 1))
    obs, info = env.reset(seed=cfg.seed + episode_index)
    if cfg.randomize_camera:
        randomize_camera(env, rng, cfg.camera_pos_noise)
        obs = env.unwrapped.get_obs()
    expert = None
    failures_before = 0 if planner is None else int(planner.plan_failures)
    if source != "explore":
        if planner is None:
            raise RuntimeError("expert collection requires a planner")
        expert = ExpertPolicy(require_task(cfg.env_id).expert, planner, verbose=cfg.verbose)
        expert.reset(env)
    rows = []
    success = _flag(info, "success")
    goal_source = "unset"
    for _ in range(cfg.max_steps):
        if source == "explore":
            action = _explore_action(env, rng, cfg)
        else:
            action = expert.act(env)
            if source == "noisy":
                action = _noisy(action, rng, cfg.noise_epsilon)
        snap = _snapshot(obs, env, cfg, groups, rng)
        goal_source = snap["goal_source"]
        rows.append((snap, action.astype(np.float32), success, False))
        obs, _reward, terminated, truncated, info = env.step(action)
        success = _flag(info, "success")
        # Success sets `terminated`, but the physics state is still valid and
        # the tier budgets are ~50/150/200 control steps. Keep stepping until
        # the time limit so short successes still yield level-2 windows.
        del terminated
        if _as_bool(truncated):
            snap = _snapshot(obs, env, cfg, groups, rng)
            rows.append((snap, hold_action(0.0), success, True))
            break
    else:
        snap = _snapshot(obs, env, cfg, groups, rng)
        rows.append((snap, hold_action(0.0), success, True))
    episode = {
        "points": np.stack([row[0]["points"] for row in rows]),
        "actions": np.stack([row[1] for row in rows]),
        "goal_coords": np.stack([row[0]["goal"] for row in rows]),
        "robot_state": np.stack([row[0]["robot_state"] for row in rows]),
        "point_count": np.array([row[0]["point_count"] for row in rows], dtype=np.int32),
        "success": np.array([row[2] for row in rows], dtype=np.bool_),
        "done": np.array([row[3] for row in rows], dtype=np.bool_),
        "source": SOURCE_CODE[source],
        "episode_success": bool(rows[-1][2]),
        "plan_failures": 0 if planner is None else int(planner.plan_failures) - failures_before,
        "goal_source": goal_source,
    }
    if cfg.store_rgb:
        episode["rgb"] = np.stack([row[0]["rgb"] for row in rows])
    return episode


def _apply_task_defaults(cfg: CollectConfig) -> CollectConfig:
    spec = require_task(cfg.env_id)
    if cfg.max_steps is None:
        cfg.max_steps = spec.max_steps
    return cfg


def _configure_planner(planner, cfg: CollectConfig) -> float:
    resolution = resolve_mesh_sampling_resolution(cfg.env_id, cfg.mesh_sampling_resolution)
    planner.mesh_sampling_resolution = resolution
    return resolution


def _open_collection_env(cfg: CollectConfig, sim_backend: str):
    env = make_env(cfg, sim_backend, num_envs=1)
    spec = require_task(cfg.env_id)
    env._point_hjepa_actor_attrs = spec.actor_attrs
    return env


def collect(cfg: CollectConfig) -> dict:
    ensure_vulkan()
    cfg = _apply_task_defaults(cfg)
    sources = assign_sources(cfg.num_episodes, cfg.source_weights, cfg.seed)
    requested = cfg.sim_backend
    backend = resolve_sim_backend(requested)
    needs_planner = any(source != "explore" for source in sources)
    notes = []
    if backend == "gpu" and needs_planner:
        if requested == "auto":
            notes.append(
                "CUDA is available, but SRMP expert collection reads a CPU SAPIEN "
                "scene. auto selected CPU simulation plus workers instead of the "
                "GPU backend. mplib was not substituted."
            )
            backend = "cpu"
        else:
            raise SrmpRequiresCpuScene(
                "sim_backend=gpu cannot host the SRMP expert. SRMP imports a CPU "
                "SAPIEN scene (read_sim / joint state); ManiSkill's GPU backend "
                "does not expose one per parallel env. Refusing to switch to mplib. "
                "Use sim_backend=cpu for expert or noisy episodes. Random "
                "exploration alone can use the GPU backend."
            )
    if backend == "gpu":
        return _collect_gpu_explore(cfg, sources, notes)
    return _collect_cpu(cfg, sources, notes)


def _collect_cpu(cfg: CollectConfig, sources: list[str], notes: list[str]) -> dict:
    probe = {}
    if cfg.probe_srmp_issues and cfg.planner == "srmp" and any(s != "explore" for s in sources):
        probe = _run_probe(cfg)
    jobs = list(enumerate(sources))
    started = time.perf_counter()
    if cfg.num_workers <= 1:
        episodes = _run_serial(cfg, jobs)
    else:
        episodes = _run_workers(cfg, jobs)
    elapsed = time.perf_counter() - started
    return _finish(cfg, episodes, sources, probe, notes, elapsed)


def _run_probe(cfg: CollectConfig) -> dict:
    from point_hjepa.data.planners.srmp_backend import probe_known_issues

    env = _open_collection_env(cfg, "cpu")
    try:
        env.reset(seed=cfg.seed)
        started = time.perf_counter()
        report = probe_known_issues(env)
        report["probe_seconds"] = time.perf_counter() - started
        return report
    finally:
        env.close()


def _run_serial(cfg: CollectConfig, jobs: list[tuple[int, str]]) -> list[dict]:
    env = _open_collection_env(cfg, "cpu")
    planner = create_planner(cfg.planner)
    _configure_planner(planner, cfg)
    groups = None
    episodes = []
    try:
        for index, source in jobs:
            if groups is None:
                # Segmentation ids are stable for a fixed scene. Grab them after reset.
                env.reset(seed=cfg.seed + index)
                groups = segmentation_groups(env, env._point_hjepa_actor_attrs)
            episode = run_episode(env, planner, cfg, index, source, groups)
            episodes.append(episode)
            print(
                    f"episode {index + 1}/{cfg.num_episodes} source={source} "
                    f"frames={episode['points'].shape[0]} success={episode['episode_success']} "
                    f"plan_failures={episode['plan_failures']}",
                    flush=True,
                )
    finally:
        planner.close()
        env.close()
    return episodes


def _run_workers(cfg: CollectConfig, jobs: list[tuple[int, str]]) -> list[dict]:
    ctx = mp.get_context("spawn")
    workers = min(cfg.num_workers, len(jobs))
    chunks: list[list[tuple[int, str]]] = [[] for _ in range(workers)]
    for i, job in enumerate(jobs):
        chunks[i % workers].append(job)
    queue = ctx.Queue()
    processes = []
    for chunk in chunks:
        if not chunk:
            continue
        proc = ctx.Process(target=_worker_main, args=(cfg, chunk, queue))
        proc.start()
        processes.append(proc)
    episodes = []
    finished = 0
    while finished < len(processes):
        message = queue.get()
        kind = message[0]
        if kind == "episode":
            episodes.append(message[1])
            print(
                f"episode {message[1]['_index'] + 1}/{cfg.num_episodes} "
                f"source={message[2]} frames={message[1]['points'].shape[0]} "
                f"success={message[1]['episode_success']}",
                flush=True,
            )
        elif kind == "error":
            for proc in processes:
                proc.terminate()
            raise RuntimeError(message[1])
        elif kind == "done":
            finished += 1
    for proc in processes:
        proc.join()
    episodes.sort(key=lambda episode: episode["_index"])
    return episodes


def _worker_main(cfg: CollectConfig, jobs: list[tuple[int, str]], queue) -> None:
    try:
        ensure_vulkan()
        env = _open_collection_env(cfg, "cpu")
        planner = create_planner(cfg.planner)
        _configure_planner(planner, cfg)
        groups = None
        for index, source in jobs:
            if groups is None:
                env.reset(seed=cfg.seed + index)
                groups = segmentation_groups(env, env._point_hjepa_actor_attrs)
            episode = run_episode(env, planner, cfg, index, source, groups)
            episode["_index"] = index
            queue.put(("episode", episode, source))
        planner.close()
        env.close()
        queue.put(("done",))
    except Exception:
        queue.put(("error", traceback.format_exc()))


def _collect_gpu_explore(cfg: CollectConfig, sources: list[str], notes: list[str]) -> dict:
    if any(source != "explore" for source in sources):
        raise SrmpRequiresCpuScene("GPU collection is implemented for random exploration only.")
    num_envs = max(1, cfg.num_envs)
    env = make_env(cfg, "gpu", num_envs=num_envs)
    spec = require_task(cfg.env_id)
    env._point_hjepa_actor_attrs = spec.actor_attrs
    started = time.perf_counter()
    episodes = []
    try:
        cursor = 0
        while cursor < len(sources):
            batch = sources[cursor : cursor + num_envs]
            obs, info = env.reset(seed=cfg.seed + cursor)
            groups = segmentation_groups(env, spec.actor_attrs)
            buffers = [[] for _ in batch]
            success = np.asarray(info.get("success", [False] * len(batch))).reshape(-1)
            rngs = [np.random.default_rng(cfg.seed + 10007 * (cursor + i + 1)) for i in range(len(batch))]
            done = np.zeros(len(batch), dtype=bool)
            for _ in range(cfg.max_steps):
                actions = np.stack([_explore_action(env, rngs[i], cfg) for i in range(len(batch))])
                # Vectorized GPU envs step a whole batch. Per-env TCP biasing above
                # uses env 0's state for every row when the pose tensor is batched;
                # the action is still a valid workspace sample and failures are kept.
                for i in range(len(batch)):
                    if done[i]:
                        actions[i] = 0.0
                        continue
                    snap = _snapshot_env(obs, env, cfg, groups, rngs[i], i)
                    buffers[i].append((snap, actions[i].astype(np.float32), bool(success[i]), False))
                obs, _reward, terminated, truncated, info = env.step(actions)
                success = np.asarray(info.get("success", np.zeros(len(batch)))).reshape(-1)
                finished = np.asarray(truncated).reshape(-1)
                for i in range(len(batch)):
                    if done[i]:
                        continue
                    if bool(finished[i]) or _ == cfg.max_steps - 1:
                        snap = _snapshot_env(obs, env, cfg, groups, rngs[i], i)
                        buffers[i].append((snap, np.zeros(7, np.float32), bool(success[i]), True))
                        done[i] = True
            for i, source in enumerate(batch):
                rows = buffers[i]
                episodes.append(_rows_to_episode(rows, source, cfg, index=cursor + i))
            cursor += len(batch)
    finally:
        env.close()
    elapsed = time.perf_counter() - started
    notes.append("Collected random exploration on the GPU backend.")
    return _finish(cfg, episodes, sources, {}, notes, elapsed)


def _snapshot_env(obs, env, cfg, groups, rng, index: int) -> dict:
    cloud = obs["pointcloud"]
    pre = preprocess_pointclouds(
        cloud["xyzw"][index],
        cloud["segmentation"][index],
        rgb=cloud["rgb"][index] if cfg.store_rgb else None,
        base_p=vec3(env.unwrapped.agent.robot.pose.p[index]),
        base_q=quat4(env.unwrapped.agent.robot.pose.q[index]),
        robot_ids=groups[0],
        object_ids=groups[1],
        table_ids=groups[2],
        config=PreprocessConfig(
            num_points=cfg.num_points,
            workspace_min=tuple(cfg.workspace_min),
            workspace_max=tuple(cfg.workspace_max),
            table_keep=cfg.table_keep,
            downsample=cfg.downsample,
            voxel_size=cfg.voxel_size,
            store_rgb=cfg.store_rgb,
        ),
        rng=rng,
    )
    goal, goal_source = _goal_vector(obs, env, cfg.env_id)
    qpos = np.asarray(obs["agent"]["qpos"][index].detach().cpu().numpy(), dtype=np.float32).reshape(-1)
    qvel = np.asarray(obs["agent"]["qvel"][index].detach().cpu().numpy(), dtype=np.float32).reshape(-1)
    tcp = np.concatenate(
        [vec3(env.unwrapped.agent.tcp.pose.p[index]), quat4(env.unwrapped.agent.tcp.pose.q[index])]
    ).astype(np.float32)
    row = {
        "points": pre["points"][0],
        "point_count": int(pre["point_count"][0]),
        "goal": goal,
        "goal_source": goal_source,
        "robot_state": np.concatenate([qpos, qvel, tcp]),
    }
    if cfg.store_rgb:
        row["rgb"] = pre["rgb"][0]
    return row


def _rows_to_episode(rows, source: str, cfg: CollectConfig, index: int) -> dict:
    episode = {
        "points": np.stack([row[0]["points"] for row in rows]),
        "actions": np.stack([row[1] for row in rows]),
        "goal_coords": np.stack([row[0]["goal"] for row in rows]),
        "robot_state": np.stack([row[0]["robot_state"] for row in rows]),
        "point_count": np.array([row[0]["point_count"] for row in rows], dtype=np.int32),
        "success": np.array([row[2] for row in rows], dtype=np.bool_),
        "done": np.array([row[3] for row in rows], dtype=np.bool_),
        "source": SOURCE_CODE[source],
        "episode_success": bool(rows[-1][2]),
        "plan_failures": 0,
        "goal_source": rows[-1][0]["goal_source"],
        "_index": index,
    }
    if cfg.store_rgb:
        episode["rgb"] = np.stack([row[0]["rgb"] for row in rows])
    return episode


def _finish(cfg, episodes, sources, probe, notes, elapsed: float) -> dict:
    episodes = sorted(episodes, key=lambda episode: episode.get("_index", 0))
    goal_source = episodes[0].get("goal_source", "") if episodes else ""
    writer = AsyncEpisodeWriter(cfg.out, cfg.format)
    writer.set_attrs(
        {
            "env_id": cfg.env_id,
            "obs_mode": cfg.obs_mode,
            "control_mode": cfg.control_mode,
            "planner": cfg.planner,
            "num_points": cfg.num_points,
            "store_rgb": cfg.store_rgb,
            "downsample": cfg.downsample,
            "source_names": ["expert", "noisy", "explore"],
            "source_weights": cfg.source_weights,
            "noise_epsilon": cfg.noise_epsilon,
            "context_frames": cfg.context_frames,
            "frameskip": cfg.frameskip,
            "horizon": cfg.horizon,
            "level2_stride": cfg.level2_stride,
            "level2_horizon": cfg.level2_horizon,
            "workspace_min": list(cfg.workspace_min),
            "workspace_max": list(cfg.workspace_max),
            "goal_source": goal_source,
            "robot_state_layout": "qpos, qvel, tcp_xyz, tcp_quat_wxyz",
            "action_layout": "normalized pd_ee_delta_pose (dx, dy, dz, dex, dey, dez, gripper); action[t] is applied at frame t; the terminal frame's action is padding",
            "srmp_workarounds": {
                "srmp": ">=0.3.0",
                "uses_set_base_pose": True,
                "base_contact_allowed": True,
                "validity_backend": "coal",
                "post_grasp_lift_m": 0.05,
                "mesh_sampling_resolution": resolve_mesh_sampling_resolution(
                    cfg.env_id, cfg.mesh_sampling_resolution
                ),
                "probe": probe,
            },
            "notes": notes,
        }
    )
    for episode in episodes:
        episode.pop("_index", None)
        episode.pop("goal_source", None)
        writer.write_episode(episode)
    written = writer.close()
    path = Path(cfg.out)
    nbytes = _tree_size(path)
    frames = sum(int(episode["points"].shape[0]) for episode in episodes)
    successes = {
        name: int(sum(episode["episode_success"] and episode["source"] == code for episode in episodes))
        for name, code in SOURCE_CODE.items()
    }
    dataset = PointCloudWindowDataset(
        path,
        context_frames=cfg.context_frames,
        frameskip=cfg.frameskip,
        horizon=cfg.horizon,
        level2_stride=cfg.level2_stride,
        level2_horizon=cfg.level2_horizon,
    )
    window_count = len(dataset)
    sample_shapes = {}
    if window_count:
        sample = dataset[0]
        sample_shapes = {key: list(value.shape) for key, value in sample.items()}
        _assert_window_inside_episode(dataset, sample)
    dataset.close()
    report = {
        "env_id": cfg.env_id,
        "planner": cfg.planner,
        "episodes": len(episodes),
        "frames": frames,
        "source_counts": source_counts(sources),
        "success_by_source": successes,
        "success_total": int(sum(episode["episode_success"] for episode in episodes)),
        "plan_failures": int(sum(episode["plan_failures"] for episode in episodes)),
        "wall_seconds": elapsed,
        "steps_per_second": frames / elapsed if elapsed else None,
        "episodes_per_second": len(episodes) / elapsed if elapsed else None,
        "disk_bytes": nbytes,
        "disk_mib": nbytes / (1024 * 1024),
        "path": str(path),
        "windows": window_count,
        "sample_shapes": sample_shapes,
        "goal_source": goal_source,
        "srmp_probe": probe,
        "notes": notes,
        "writer": written,
    }
    report_path = path.parent / f"{path.name}_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in (
        "episodes", "frames", "source_counts", "success_by_source", "success_total",
        "plan_failures", "wall_seconds", "steps_per_second", "disk_mib", "windows", "goal_source",
    )}, indent=2), flush=True)
    return report


def _assert_window_inside_episode(dataset: PointCloudWindowDataset, sample: dict) -> None:
    episode = int(sample["episode_index"])
    start, end = dataset.episode_span(episode)
    # Re-derive the anchor from the stored index of this first sample.
    _episode, anchor = dataset._index[0]
    assert _episode == episode
    length = end - start
    from point_hjepa.data.dataset import level_indices

    for stride, horizon in (
        (dataset.frameskip, dataset.horizon),
        (dataset.level2_stride, dataset.level2_horizon),
    ):
        ctx, fut = level_indices(anchor, dataset.context_frames, stride, horizon)
        assert min(ctx + fut) >= 0
        assert max(ctx + fut) < length


def _tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect 3D Point H-JEPA point-cloud episodes")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--planner", default=None, choices=["srmp", "mplib"])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--sim-backend", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(
        args.config,
        {
            "out": args.out,
            "num_episodes": args.num_episodes,
            "planner": args.planner,
            "num_workers": args.workers,
            "sim_backend": args.sim_backend,
            "seed": args.seed,
            "verbose": True if args.verbose else None,
        },
    )
    if cfg.env_id not in TASKS and cfg.env_id != "SetTable-v1":
        pass
    try:
        collect(cfg)
    except Exception as exc:
        if exc.__class__.__name__ in ("TaskUnavailable", "SrmpRequiresCpuScene"):
            raise SystemExit(str(exc)) from exc
        raise


if __name__ == "__main__":
    main()
