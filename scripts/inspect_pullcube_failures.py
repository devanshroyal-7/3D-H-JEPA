"""Find, classify, and replay failed SRMP expert episodes in ManiSkill3's SAPIEN GUI.

By default, scans ``PullCubeTool-v1`` seeds headlessly to locate failed SRMP
expert rollouts, prints per-stage failure telemetry, and opens ManiSkill3's
native SAPIEN 3D viewer (``render_mode="human"``) to replay the failures live:

    python scripts/inspect_pullcube_failures.py
    python scripts/inspect_pullcube_failures.py --seeds 500003 500012 --loop
    python scripts/inspect_pullcube_failures.py --no-human --save-mp4-dir outputs/fail_mp4
    python scripts/inspect_pullcube_failures.py --zarr-root outputs/datasets/v1 --out-dir outputs/fail_viz
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from point_hjepa.data.collect_maniskill3_6d import (
    ARM_DELTA_SCALE,
    SRMPExpertController,
    extract_step_metadata,
    joint_target_to_delta_action,
)

SUCCESS_RADIUS_M = 0.60  # PullCubeToolEnv.evaluate(): ||cube_xy - base_xy|| < 0.60


@dataclass
class StageLog:
    name: str
    step_idx: int
    is_grasped: bool
    yaw_slip_deg: float
    cube_pos: list[float]


@dataclass
class EpisodeOutcome:
    seed: int
    success: bool
    reason: str
    steps: int
    dist_start: float
    dist_end: float
    stage_logs: list[StageLog]
    frames: list[np.ndarray]
    closed: bool = False


def _quat_yaw_deg(q_wxyz: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in q_wxyz]
    return float(np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))))


def _wrap_deg(deg: float) -> float:
    return float((deg + 180.0) % 360.0 - 180.0)


def _target_actor(env: Any, task_id: str) -> Any:
    u = env.unwrapped
    if task_id == "PullCubeTool-v1":
        return u.l_shape_tool
    if task_id == "StackCube-v1":
        return u.cubeA
    raise ValueError(f"Unsupported task_id: {task_id!r}")


def _render_and_pace(env: Any, dt: float) -> bool:
    viewer = env.render()
    if dt > 0:
        time.sleep(dt)
    return bool(getattr(viewer, "closed", False))


def _pause_viewer(env: Any, seconds: float) -> bool:
    tick = 0.05
    for _ in range(max(1, int(round(seconds / tick)))):
        if _render_and_pace(env, tick):
            return True
    return False


def _grab_rgb_frame(env: Any) -> np.ndarray:
    frame = env.render()
    if isinstance(frame, torch.Tensor):
        frame = frame[0].detach().cpu().numpy()
    elif isinstance(frame, (list, tuple)):
        frame = np.asarray(frame[0])
    if frame.ndim == 4:
        frame = frame[0]
    return frame.astype(np.uint8, copy=False)


def classify_failure(
    ach_traj: np.ndarray,
    max_steps: int,
    stage_logs: list[StageLog] | None = None,
    plan_failed: bool = False,
    dropped_tool: bool = False,
) -> str:
    """Categorize why a PullCubeTool-v1 (or StackCube-v1) SRMP episode failed."""
    if plan_failed:
        return "plan_failed"
    if dropped_tool:
        return "dropped_tool"
    if len(ach_traj) == 0:
        return "empty_episode"

    if stage_logs:
        post_grasp = [s for s in stage_logs if s.name in {"lift", "transport", "lower_hook", "place_or_pull"}]
        if post_grasp and not any(s.is_grasped for s in post_grasp):
            return "dropped_tool"

    T = len(ach_traj)
    d0 = float(np.linalg.norm(ach_traj[0, :2]))
    d_end = float(np.linalg.norm(ach_traj[-1, :2]))
    d_min = float(np.linalg.norm(ach_traj[:, :2], axis=-1).min())
    disp = float(np.linalg.norm(ach_traj[-1, :2] - ach_traj[0, :2]))
    max_disp = float(np.linalg.norm(ach_traj[:, :2] - ach_traj[0, :2], axis=-1).max())

    if d_min < SUCCESS_RADIUS_M and d_end >= SUCCESS_RADIUS_M:
        return "succeeded_then_pushed_back"
    if T >= max_steps and d_end < d0 - 0.05:
        return "hit_step_cap_mid_pull"
    if max_disp < 0.03:
        return "missed_cube"
    if d_end > d0 + 0.03:
        return "pushed_cube_away"
    if disp < 0.15:
        return "cube_slipped_off_hook"
    return "pull_too_short"


def _run_expert_episode(
    env: Any,
    expert: SRMPExpertController,
    task_id: str,
    seed: int,
    max_steps: int,
    render_mode: str | None,
    dt: float,
) -> EpisodeOutcome:
    """Execute one SRMP expert rollout mirroring collect_maniskill3_6d.py."""
    u = env.unwrapped
    device = u.device
    env.reset(seed=seed)

    grasp_actor = _target_actor(env, task_id)
    yaw0 = _quat_yaw_deg(grasp_actor.pose.q[0].detach().cpu().numpy())
    ach0 = extract_step_metadata(env, task_id)["achieved_point"][0].detach().cpu().numpy()
    ach_list: list[np.ndarray] = [ach0]
    frames: list[np.ndarray] = []

    stages = expert.plan_episode()
    if stages is None:
        d0 = float(np.linalg.norm(ach0[:2]))
        return EpisodeOutcome(
            seed=seed,
            success=False,
            reason="plan_failed",
            steps=0,
            dist_start=d0,
            dist_end=d0,
            stage_logs=[],
            frames=frames,
        )

    closed = False
    if render_mode == "human":
        print(f"\n=== Seed {seed} ({task_id}) ===", flush=True)
        if _render_and_pace(env, dt):
            return EpisodeOutcome(seed, False, "viewer_closed", 0, 0.0, 0.0, [], frames, closed=True)
    elif render_mode == "rgb_array":
        frames.append(_grab_rgb_frame(env))

    step_idx = 0
    stage_logs: list[StageLog] = []

    for stage in stages:
        if render_mode == "human":
            print(f"  [step {step_idx:3d}] stage -> {stage.name}", flush=True)

        targets = list(stage.waypoints) + [stage.waypoints[-1]] * stage.settle_steps

        def _step_to(wp_target: np.ndarray) -> bool:
            nonlocal step_idx, closed
            if step_idx >= max_steps or closed:
                return True
            q_curr = u.agent.robot.get_qpos()[0, :7].detach().cpu().numpy()
            act_np = joint_target_to_delta_action(q_curr, wp_target, stage.gripper, ARM_DELTA_SCALE)
            act_t = torch.from_numpy(act_np).unsqueeze(0).to(device)
            _, _, term, trunc, _ = env.step(act_t)
            step_idx += 1

            meta = extract_step_metadata(env, task_id)
            ach_list.append(meta["achieved_point"][0].detach().cpu().numpy())

            if render_mode == "human":
                if _render_and_pace(env, dt):
                    closed = True
                    return True
            elif render_mode == "rgb_array":
                frames.append(_grab_rgb_frame(env))
            return bool(term[0] or trunc[0])

        done = False
        max_wp_tries = 3 if stage.name == "place_or_pull" else 1
        for wp in targets:
            for _ in range(max_wp_tries):
                done = _step_to(wp)
                if done:
                    break
                q_curr = u.agent.robot.get_qpos()[0, :7].detach().cpu().numpy()
                if float(np.max(np.abs(q_curr - wp))) <= 0.035:
                    break
            if done:
                break

        if not done and stage.name in {"grasp", "lower_hook", "place_or_pull"}:
            final_wp = stage.waypoints[-1]
            for _ in range(8):
                q_curr = u.agent.robot.get_qpos()[0, :7].detach().cpu().numpy()
                if float(np.max(np.abs(q_curr - final_wp))) < 0.015:
                    break
                done = _step_to(final_wp)
                if done:
                    break

        is_grasped = bool(u.agent.is_grasping(grasp_actor)[0])
        yaw_now = _quat_yaw_deg(grasp_actor.pose.q[0].detach().cpu().numpy())
        yaw_slip = _wrap_deg(yaw_now - yaw0)
        stage_logs.append(
            StageLog(
                name=stage.name,
                step_idx=step_idx,
                is_grasped=is_grasped,
                yaw_slip_deg=yaw_slip,
                cube_pos=[round(float(v), 3) for v in ach_list[-1]],
            )
        )
        if done or step_idx >= max_steps or closed:
            break

    ev = u.evaluate()
    success = bool(ev.get("success", torch.tensor([False]))[0])
    ach_traj = np.stack(ach_list, axis=0)
    d0 = float(np.linalg.norm(ach_traj[0, :2]))
    d_end = float(np.linalg.norm(ach_traj[-1, :2]))
    reason = "success" if success else classify_failure(ach_traj, max_steps, stage_logs=stage_logs)

    if render_mode == "human":
        print(
            f"  => Result: {reason} | steps={step_idx} | dist_xy={d0:.3f}m -> {d_end:.3f}m",
            flush=True,
        )

    return EpisodeOutcome(
        seed=seed,
        success=success,
        reason=reason,
        steps=step_idx,
        dist_start=d0,
        dist_end=d_end,
        stage_logs=stage_logs,
        frames=frames,
        closed=closed,
    )


def _print_outcome_summary(outcomes: list[EpisodeOutcome], task_id: str) -> None:
    total = len(outcomes)
    n_succ = sum(1 for o in outcomes if o.success)
    fails = [o for o in outcomes if not o.success]
    print(f"\n[{task_id}] Tested {total} seeds: {n_succ} succeeded ({100.0 * n_succ / max(1, total):.1f}%), {len(fails)} failed")
    counts = Counter(o.reason for o in fails)
    for reason, cnt in counts.most_common():
        print(f"  {reason:28s}: {cnt:3d} ({100.0 * cnt / max(1, len(fails)):5.1f}% of failures)")

    for o in fails:
        print(f"\nFAIL seed={o.seed} reason={o.reason} steps={o.steps} dist_xy={o.dist_start:.3f}m -> {o.dist_end:.3f}m")
        for s in o.stage_logs:
            print(
                f"  {s.name:14s} step={s.step_idx:3d} grasped={str(s.is_grasped):5s} "
                f"yaw_slip={s.yaw_slip_deg:+5.1f}deg cube={s.cube_pos}"
            )


def _analyze_zarr_shards(root: Path, out_dir: Path, max_plots: int) -> None:
    """Offline CPU classification + XY trajectory plot from sharded Zarr stores."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import zarr

    stores = sorted(root.glob("PullCubeTool-v1/shard_*/PullCubeTool-v1_expert_srmp.zarr"))
    if not stores:
        raise FileNotFoundError(f"No PullCubeTool-v1_expert_srmp.zarr stores found under {root}")

    counts: Counter[str] = Counter()
    total_eps = 0
    failed_records: list[tuple[str, int, str, int, float, np.ndarray, np.ndarray, np.ndarray]] = []

    for store in stores:
        g = zarr.open_group(str(store), mode="r")
        ends = np.asarray(g["episode_ends"], dtype=np.int64)
        starts = np.concatenate([[0], ends[:-1]])
        ach_all = np.asarray(g["achieved_point"], dtype=np.float32)
        ee_all = np.asarray(g["ee_point"], dtype=np.float32)
        prop_all = np.asarray(g["proprio"], dtype=np.float32)

        for ep_i, (s, e) in enumerate(zip(starts, ends)):
            total_eps += 1
            ach = ach_all[s:e]
            ee = ee_all[s:e]
            grip_w = prop_all[s:e, -1]
            d_end = float(np.linalg.norm(ach[-1, :2]))
            if d_end < SUCCESS_RADIUS_M:
                continue
            dropped = len(grip_w) > 10 and float(grip_w[len(grip_w) // 2 :].min()) < 0.015
            reason = classify_failure(ach, max_steps=160, dropped_tool=dropped)
            counts[reason] += 1
            failed_records.append((store.parent.name, ep_i, reason, len(ach), d_end, ach, ee, grip_w))

    n_fail = len(failed_records)
    print(f"Total episodes: {total_eps} | Success: {total_eps - n_fail} ({100 * (total_eps - n_fail) / total_eps:.1f}%) | Failed: {n_fail}")
    for k, v in counts.most_common():
        print(f"  {k:28s}: {v:4d} ({100 * v / max(1, n_fail):5.1f}% of failures)")

    out_dir.mkdir(parents=True, exist_ok=True)
    n_plot = min(max_plots, n_fail)
    if n_plot > 0:
        cols = 4
        rows = (n_plot + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
        theta = np.linspace(-np.pi / 2, np.pi / 2, 120)
        for idx in range(rows * cols):
            ax = axes[idx // cols][idx % cols]
            if idx >= n_plot:
                ax.axis("off")
                continue
            shard, ep_i, reason, T, d_end, ach, ee, grip_w = failed_records[idx]
            ax.plot(SUCCESS_RADIUS_M * np.cos(theta), SUCCESS_RADIUS_M * np.sin(theta), "g--", lw=1, label="r=0.60m")
            ax.plot(ee[:, 0], ee[:, 1], color="#2a78d6", lw=1.2, alpha=0.8, label="TCP")
            ax.plot(ach[:, 0], ach[:, 1], color="#eb6834", lw=2.0, label="Cube")
            ax.scatter(ach[0, 0], ach[0, 1], c="#eb6834", s=25, marker="o")
            ax.scatter(ach[-1, 0], ach[-1, 1], c="red", s=40, marker="x")
            ax.set_xlim(0.0, 1.05)
            ax.set_ylim(-0.45, 0.45)
            ax.set_aspect("equal")
            ax.set_title(
                f"{shard} ep{ep_i} | {reason}\nT={T}, d_end={d_end:.2f}m, grip_min={grip_w[len(grip_w) // 2 :].min():.3f}",
                fontsize=8,
            )
            if idx == 0:
                ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        out_png = out_dir / "pullcube_failed_trajectories.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Saved trajectory grid -> {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="PullCubeTool-v1", choices=["PullCubeTool-v1", "StackCube-v1"])
    parser.add_argument("--start-seed", type=int, default=500000, help="First seed to scan (PullCubeTool shard_0 starts at 500000).")
    parser.add_argument("--max-seeds", type=int, default=80, help="Max seeds to scan when searching for failures.")
    parser.add_argument("--max-failures", type=int, default=10, help="Stop scanning once this many execution failures are found.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Explicit seeds to inspect instead of scanning.")
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--sim-backend", default="physx_cuda", choices=["physx_cuda", "cpu"])
    parser.add_argument("--fps", type=float, default=20.0, help="Playback FPS in the SAPIEN GUI viewer.")
    parser.add_argument("--pause", type=float, default=1.2, help="Pause seconds between failed episodes in viewer.")
    parser.add_argument("--loop", action="store_true", help="Loop failed episodes continuously in the GUI viewer.")
    parser.add_argument("--no-human", action="store_true", help="Skip opening the interactive SAPIEN GUI viewer.")
    parser.add_argument("--save-mp4-dir", type=Path, default=None, help="Save MP4 videos of failed seeds to this directory.")
    parser.add_argument("--zarr-root", type=Path, default=None, help="If set, run offline CPU analysis on Zarr shards instead of sim.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/fail_viz"), help="Output directory for --zarr-root plots.")
    parser.add_argument("--max-plots", type=int, default=16, help="Max episodes to plot in --zarr-root mode.")
    args = parser.parse_args()

    if args.zarr_root is not None:
        _analyze_zarr_shards(args.zarr_root, args.out_dir, args.max_plots)
        return

    # 1. Headless pass to locate failed execution seeds quickly (or evaluate explicit --seeds).
    scan_seeds = list(args.seeds) if args.seeds is not None else [args.start_seed + i for i in range(args.max_seeds)]
    env = gym.make(
        args.task,
        num_envs=1,
        obs_mode="none",
        control_mode="pd_joint_delta_pos",
        sim_backend=args.sim_backend,
        render_mode=None,
        max_episode_steps=args.max_steps,
    )
    outcomes: list[EpisodeOutcome] = []
    failed_seeds: list[int] = []
    try:
        expert = SRMPExpertController(env, args.task)
        for seed in scan_seeds:
            out = _run_expert_episode(env, expert, args.task, seed, args.max_steps, render_mode=None, dt=0.0)
            outcomes.append(out)
            if not out.success and out.reason != "plan_failed":
                failed_seeds.append(seed)
                if args.seeds is None and len(failed_seeds) >= args.max_failures:
                    break
    finally:
        env.close()

    _print_outcome_summary(outcomes, args.task)
    if args.seeds is not None:
        replay_seeds = list(args.seeds)
    else:
        replay_seeds = failed_seeds

    if not replay_seeds:
        print("\nNo failed episodes to replay.")
        return

    # 2. Optional MP4 export.
    if args.save_mp4_dir is not None:
        import imageio.v2 as imageio

        args.save_mp4_dir.mkdir(parents=True, exist_ok=True)
        rec_env = gym.make(
            args.task,
            num_envs=1,
            obs_mode="none",
            control_mode="pd_joint_delta_pos",
            sim_backend=args.sim_backend,
            render_mode="rgb_array",
            max_episode_steps=args.max_steps,
        )
        try:
            expert = SRMPExpertController(rec_env, args.task)
            for seed in replay_seeds:
                out = _run_expert_episode(rec_env, expert, args.task, seed, args.max_steps, render_mode="rgb_array", dt=0.0)
                mp4_path = args.save_mp4_dir / f"{args.task}_seed{seed}_{out.reason}.mp4"
                imageio.mimsave(str(mp4_path), out.frames, fps=int(round(args.fps)))
                print(f"Saved {mp4_path} ({len(out.frames)} frames)", flush=True)
        finally:
            rec_env.close()

    # 3. Interactive ManiSkill3 SAPIEN 3D viewer replay.
    if args.no_human:
        return

    dt = 1.0 / max(1.0, float(args.fps))
    gui_env = gym.make(
        args.task,
        num_envs=1,
        obs_mode="none",
        control_mode="pd_joint_delta_pos",
        sim_backend=args.sim_backend,
        render_mode="human",
        max_episode_steps=args.max_steps,
    )
    try:
        expert = SRMPExpertController(gui_env, args.task)
        while True:
            for i, seed in enumerate(replay_seeds):
                out = _run_expert_episode(gui_env, expert, args.task, seed, args.max_steps, render_mode="human", dt=dt)
                if out.closed:
                    return
                if (i < len(replay_seeds) - 1 or args.loop) and _pause_viewer(gui_env, args.pause):
                    return
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\nStopped failure viewer.", flush=True)
    finally:
        gui_env.close()


if __name__ == "__main__":
    main()
