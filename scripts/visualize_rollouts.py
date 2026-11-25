"""Visual and quantitative rollout verifier for ManiSkill3 6D data collection.

Renders side-by-side MP4 videos (RGB camera | preprocessed 6D point cloud with
ee_point & achieved_point | 8D pd_joint_delta_pos action bars + telemetry) for
all 6 trajectory categories on StackCube-v1 and PullCubeTool-v1, plus filmstrip
PNG contact sheets and compact quantized JSON for interactive 3D replay.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from point_hjepa.data.collect_maniskill3_6d import (
    ARM_DELTA_SCALE,
    DEFAULT_BOUNDS,
    SRMPExpertController,
    ZarrTrajectoryWriter,
    _flatten_stages,
    _sample_ou_noise,
    _squeeze_state_dict_to_cpu,
    _stack_state_dicts,
    extract_step_metadata,
    joint_target_to_delta_action,
    preprocess_pointcloud_batch,
    restore_snapshots_with_pd_sync,
)

MODE_LABELS = [
    ("expert_srmp", "Expert SRMP"),
    ("noisy_expert", "Noisy Expert (s=0.12)"),
    ("interrupted_exp_to_rand", "Interrupted (Exp->Rand)"),
    ("mid_grasp_held_ou", "Mid-Grasp Held OU"),
    ("release_drop", "Release / Drop"),
    ("random_ou", "Random OU"),
]


def _render_rgb(env: Any) -> np.ndarray:
    frame = env.render()
    if isinstance(frame, torch.Tensor):
        arr = frame[0].detach().cpu().numpy()
    else:
        arr = np.asarray(frame)
        if arr.ndim == 4:
            arr = arr[0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr[:, :, :3]


def _project_points(
    pts: np.ndarray,
    width: int,
    height: int,
    yaw_deg: float = -42.0,
    pitch_deg: float = 28.0,
    center: tuple[float, float, float] = (0.45, 0.0, 0.14),
    scale: float = 360.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthographic-perspective projection from robot-base frame to 2D image."""
    p = pts - np.array(center, dtype=np.float32)
    cy, sy = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    cp, sp = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    cam = p @ Rz.T @ Rx.T
    # cam[:, 0]=right, cam[:, 1]=depth, cam[:, 2]=up
    depth = cam[:, 1]
    persp = 1.0 / np.clip(1.8 + depth * 0.45, 0.5, 4.0)
    u = np.round(width * 0.5 + cam[:, 0] * scale * persp * 1.65).astype(np.int32)
    v = np.round(height * 0.56 - cam[:, 2] * scale * persp * 1.65).astype(np.int32)
    return u, v, depth


def _draw_pointcloud_panel(
    xyz: np.ndarray,
    rgb: np.ndarray,
    ee_pt: np.ndarray,
    ach_pt: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    size: int = 448,
) -> np.ndarray:
    canvas = np.full((size, size, 3), 18, dtype=np.uint8)

    # Draw workspace bounding box edges
    (x0, x1), (y0, y1), (z0, z1) = bounds
    corners = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    cu, cv, _ = _project_points(corners, size, size)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        cv2.line(canvas, (int(cu[a]), int(cv[a])), (int(cu[b]), int(cv[b])), (55, 62, 78), 1, cv2.LINE_AA)

    # Painter's algorithm sort (far -> near)
    u, v, depth = _project_points(xyz, size, size)
    order = np.argsort(-depth)
    u_s, v_s, rgb_s = u[order], v[order], rgb[order]
    valid = (u_s >= 1) & (u_s < size - 1) & (v_s >= 1) & (v_s < size - 1)
    u_v, v_v, c_v = u_s[valid], v_s[valid], rgb_s[valid]
    canvas[v_v, u_v] = c_v
    canvas[v_v, np.minimum(u_v + 1, size - 1)] = c_v
    canvas[np.minimum(v_v + 1, size - 1), u_v] = c_v

    # Overlay ee_point (cyan circle) & achieved_point (gold diamond)
    mu, mv, _ = _project_points(np.stack([ee_pt, ach_pt], axis=0), size, size)
    ex, ey = int(np.clip(mu[0], 6, size - 7)), int(np.clip(mv[0], 6, size - 7))
    ax, ay = int(np.clip(mu[1], 6, size - 7)), int(np.clip(mv[1], 6, size - 7))
    cv2.circle(canvas, (ex, ey), 6, (0, 235, 255), 2, cv2.LINE_AA)
    diamond = np.array([[ax, ay - 7], [ax + 7, ay], [ax, ay + 7], [ax - 7, ay]], dtype=np.int32)
    cv2.polylines(canvas, [diamond], True, (255, 215, 0), 2, cv2.LINE_AA)

    cv2.putText(canvas, "Base-Frame 6D Cloud (4096 pts)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 225, 235), 1, cv2.LINE_AA)
    cv2.circle(canvas, (18, size - 28), 5, (0, 235, 255), -1)
    cv2.putText(canvas, "ee_point", (28, size - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 235, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (112, size - 33), (122, size - 23), (255, 215, 0), -1)
    cv2.putText(canvas, "achieved_point", (128, size - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 215, 0), 1, cv2.LINE_AA)
    return canvas


def _draw_telemetry_panel(
    task_id: str,
    mode_label: str,
    step: int,
    total_steps: int,
    action: np.ndarray,
    ee_pt: np.ndarray,
    ach_pt: np.ndarray,
    disp: float,
    success: bool,
    width: int = 340,
    height: int = 448,
) -> np.ndarray:
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.putText(panel, f"{task_id}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 244, 250), 2, cv2.LINE_AA)
    cv2.putText(panel, mode_label, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (130, 195, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Step: {step:03d} / {total_steps:03d}", (14, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 215, 225), 1, cv2.LINE_AA)
    succ_col = (70, 220, 120) if success else (180, 185, 195)
    cv2.putText(panel, f"Success: {success}", (175, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, succ_col, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Cube disp: {disp*100:.1f} cm  z: {ach_pt[2]*100:.1f} cm", (14, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 215, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, f"EE base: ({ee_pt[0]:.2f}, {ee_pt[1]:.2f}, {ee_pt[2]:.2f})", (14, 127), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 235, 255), 1, cv2.LINE_AA)

    cv2.putText(panel, "8D Action (pd_joint_delta_pos)", (14, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 215, 225), 1, cv2.LINE_AA)
    names = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "GRP"]
    mid_x = 185
    half_w = 120
    for i, nm in enumerate(names):
        y = 182 + i * 31
        val = float(np.clip(action[i], -1.0, 1.0))
        cv2.putText(panel, nm, (14, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 195, 205), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (mid_x - half_w, y + 2), (mid_x + half_w, y + 18), (45, 50, 62), -1)
        cv2.line(panel, (mid_x, y + 1), (mid_x, y + 19), (120, 125, 140), 1)
        bar_x = mid_x + int(round(val * half_w))
        x_lo, x_hi = min(mid_x, bar_x), max(mid_x, bar_x)
        color = (70, 190, 255) if i < 7 else ((75, 220, 130) if val > 0 else (255, 120, 100))
        cv2.rectangle(panel, (x_lo, y + 3), (max(x_lo + 1, x_hi), y + 17), color, -1)
        cv2.putText(panel, f"{val:+.2f}", (mid_x + half_w + 6, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 205, 215), 1, cv2.LINE_AA)
    return panel


def _compose_frame(
    rgb_cam: np.ndarray,
    xyz: np.ndarray,
    rgb_pc: np.ndarray,
    ee_pt: np.ndarray,
    ach_pt: np.ndarray,
    action: np.ndarray,
    task_id: str,
    mode_label: str,
    step: int,
    total_steps: int,
    disp: float,
    success: bool,
) -> np.ndarray:
    h = 448
    cam_resized = cv2.resize(rgb_cam, (h, h), interpolation=cv2.INTER_AREA)
    cv2.putText(cam_resized, "ManiSkill3 RGB Camera", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    pc_panel = _draw_pointcloud_panel(xyz, rgb_pc, ee_pt, ach_pt, DEFAULT_BOUNDS, size=h)
    tel_panel = _draw_telemetry_panel(task_id, mode_label, step, total_steps, action, ee_pt, ach_pt, disp, success, width=352, height=h)
    return np.concatenate([cam_resized, pc_panel, tel_panel], axis=1)


def _rollout_expert(
    env: Any,
    expert: SRMPExpertController,
    task_id: str,
    max_steps: int,
    seed: int,
    render_rgb: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    device = env.unwrapped.device
    gen = torch.Generator(device=device).manual_seed(seed)
    stages = None
    for attempt in range(10):
        obs, _ = env.reset(seed=seed + attempt)
        stages = expert.plan_episode()
        if stages is not None:
            break
    if stages is None:
        raise RuntimeError(f"SRMP failed across 10 seeds for {task_id}")

    records: list[dict[str, Any]] = []
    snaps: dict[str, dict[str, Any]] = {}
    key_stages = {"approach", "grasp", "lift", "transport", "place_or_pull"}
    ach0 = None
    step_idx = 0

    for s_idx, stage in enumerate(stages):
        if stage.name in key_stages:
            rem_wps, rem_grips = _flatten_stages(stages[s_idx:])
            snaps[stage.name] = {
                "stage": stage.name,
                "is_grasped": bool(stage.is_grasped),
                "state_dict": _squeeze_state_dict_to_cpu(env.unwrapped.get_state_dict()),
                "remaining_waypoints": rem_wps,
                "remaining_grippers": rem_grips,
            }

        targets = list(stage.waypoints) + [stage.waypoints[-1]] * stage.settle_steps

        def _step_to(wp_target: np.ndarray) -> bool:
            nonlocal obs, step_idx, ach0
            if step_idx >= max_steps:
                return True
            q_curr = env.unwrapped.agent.robot.get_qpos()[0, :7].cpu().numpy()
            act_np = joint_target_to_delta_action(q_curr, wp_target, stage.gripper, ARM_DELTA_SCALE)
            act_t = torch.from_numpy(act_np).unsqueeze(0).to(device)

            xyz_b, rgb_b = preprocess_pointcloud_batch(
                obs["pointcloud"]["xyzw"],
                obs["pointcloud"]["rgb"],
                env.unwrapped.agent.robot.pose,
                bounds=DEFAULT_BOUNDS,
                num_points=4096,
                voxel_size=0.0045,
                generator=gen,
            )
            meta = extract_step_metadata(env, task_id)
            ach = meta["achieved_point"][0].cpu().numpy()
            if ach0 is None:
                ach0 = ach.copy()
            obs, _, term, trunc, _ = env.step(act_t)
            ev = env.unwrapped.evaluate()
            succ = bool(ev.get("success", torch.tensor([False]))[0])
            records.append(
                {
                    "step": step_idx,
                    "rgb_cam": _render_rgb(env) if render_rgb else None,
                    "xyz": xyz_b[0].cpu().numpy(),
                    "rgb_pc": rgb_b[0].cpu().numpy(),
                    "proprio": meta["proprio"][0].cpu().numpy(),
                    "ee": meta["ee_point"][0].cpu().numpy(),
                    "ach": ach,
                    "action": act_np.copy(),
                    "disp": float(np.linalg.norm(ach - ach0)),
                    "success": succ,
                }
            )
            step_idx += 1
            return bool(term[0] or trunc[0])

        done = False
        max_wp_tries = 3 if stage.name == "place_or_pull" else 1
        for wp in targets:
            for _ in range(max_wp_tries):
                done = _step_to(wp)
                if done:
                    break
                q_curr = env.unwrapped.agent.robot.get_qpos()[0, :7].cpu().numpy()
                if float(np.max(np.abs(q_curr - wp))) <= 0.035:
                    break
            if done:
                break
        if not done and stage.name in {"grasp", "lower_hook", "place_or_pull"}:
            final_wp = stage.waypoints[-1]
            for _ in range(8):
                q_curr = env.unwrapped.agent.robot.get_qpos()[0, :7].cpu().numpy()
                if float(np.max(np.abs(q_curr - final_wp))) < 0.015:
                    break
                done = _step_to(final_wp)
                if done:
                    break
        if done or step_idx >= max_steps:
            break

    return records, snaps


def _compute_exploratory_action(
    mode_key: str,
    t: int,
    t_split: int,
    exp_act: torch.Tensor,
    ou_state: torch.Tensor,
    gen: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Compute a single 8D exploratory action from expert/OU primitives."""
    act = torch.zeros((1, 8), device=device, dtype=torch.float32)
    if mode_key == "noisy_expert":
        act[0, :7] = torch.clamp(exp_act[0, :7] + torch.randn((7,), device=device, generator=gen) * 0.12, -1.0, 1.0)
        act[0, 7] = exp_act[0, 7]
    elif mode_key == "interrupted_exp_to_rand":
        if t < t_split:
            act[0, :7] = torch.clamp(exp_act[0, :7] + torch.randn((7,), device=device, generator=gen) * 0.10, -1.0, 1.0)
            act[0, 7] = exp_act[0, 7]
        else:
            act[0, :7] = torch.empty((7,), device=device).uniform_(-1.0, 1.0, generator=gen)
            act[0, 7] = 1.0 if torch.rand((), device=device, generator=gen) > 0.5 else -1.0
    elif mode_key == "mid_grasp_held_ou":
        act[0, :7] = torch.clamp(ou_state[0], -1.0, 1.0)
        act[0, 7] = -1.0
    elif mode_key == "release_drop":
        act[0, :7] = torch.clamp(ou_state[0] * 0.5, -1.0, 1.0)
        act[0, 7] = -1.0 if t < 12 else 1.0
    else:  # random_ou
        act[0, :7] = torch.clamp(ou_state[0], -1.0, 1.0)
        act[0, 7] = 1.0 if torch.rand((), device=device, generator=gen) > 0.5 else -1.0
    return act


def _prepare_exploratory_episode(
    env: Any,
    mode_key: str,
    snaps: dict[str, dict[str, Any]],
    max_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Reset env, restore stage snapshot if needed, and return remaining expert targets."""
    device = env.unwrapped.device
    env.reset(seed=seed)

    snap: dict[str, Any] | None = None
    if mode_key in ("noisy_expert", "interrupted_exp_to_rand"):
        snap = snaps.get("approach") or next(iter(snaps.values()))
    elif mode_key in ("mid_grasp_held_ou", "release_drop"):
        snap = snaps.get("lift") or snaps.get("grasp") or next(iter(snaps.values()))

    if snap is not None:
        restore_snapshots_with_pd_sync(env, _stack_state_dicts([snap["state_dict"]], device))
        rem_wps = snap["remaining_waypoints"]
        rem_grips = snap["remaining_grippers"]
        rollout_steps = min(max_steps, max(45, len(rem_wps) + 15)) if mode_key == "noisy_expert" else 65
    else:
        rem_wps = np.zeros((1, 7), dtype=np.float32)
        rem_grips = np.ones((1,), dtype=np.float32)
        rollout_steps = 80
    return rem_wps, rem_grips, rollout_steps


def _rollout_exploratory(
    env: Any,
    task_id: str,
    mode_key: str,
    snaps: dict[str, dict[str, Any]],
    max_steps: int,
    seed: int,
    render_rgb: bool = True,
) -> list[dict[str, Any]]:
    device = env.unwrapped.device
    gen = torch.Generator(device=device).manual_seed(seed)
    rem_wps, rem_grips, rollout_steps = _prepare_exploratory_episode(env, mode_key, snaps, max_steps, seed)

    obs = env.unwrapped.get_obs()
    ou_state = torch.zeros((1, 7), device=device, dtype=torch.float32)
    t_split = max(10, rollout_steps // 3)
    records: list[dict[str, Any]] = []
    ach0 = None

    for t in range(rollout_steps):
        xyz_b, rgb_b = preprocess_pointcloud_batch(
            obs["pointcloud"]["xyzw"],
            obs["pointcloud"]["rgb"],
            env.unwrapped.agent.robot.pose,
            bounds=DEFAULT_BOUNDS,
            num_points=4096,
            voxel_size=0.0045,
            generator=gen,
        )
        meta = extract_step_metadata(env, task_id)
        ach = meta["achieved_point"][0].cpu().numpy()
        if ach0 is None:
            ach0 = ach.copy()

        q_curr = env.unwrapped.agent.robot.get_qpos()[:, :7]
        idx = min(t, len(rem_wps) - 1)
        exp_act = joint_target_to_delta_action(
            q_curr, torch.from_numpy(rem_wps[idx : idx + 1]).to(device), float(rem_grips[idx]), ARM_DELTA_SCALE
        )
        ou_state = _sample_ou_noise(ou_state, 0.15, 0.30, gen)
        act = _compute_exploratory_action(mode_key, t, t_split, exp_act, ou_state, gen, device)

        ev = env.unwrapped.evaluate()
        succ = bool(ev.get("success", torch.tensor([False]))[0])
        records.append(
            {
                "step": t,
                "rgb_cam": _render_rgb(env) if render_rgb else None,
                "xyz": xyz_b[0].cpu().numpy(),
                "rgb_pc": rgb_b[0].cpu().numpy(),
                "proprio": meta["proprio"][0].cpu().numpy(),
                "ee": meta["ee_point"][0].cpu().numpy(),
                "ach": ach,
                "action": act[0].cpu().numpy().copy(),
                "disp": float(np.linalg.norm(ach - ach0)),
                "success": succ,
            }
        )
        obs, _, _, _, _ = env.step(act)

    return records


def _save_video_and_filmstrip_row(
    records: list[dict[str, Any]],
    task_id: str,
    mode_key: str,
    mode_label: str,
    mp4_path: Path,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(mp4_path), fps=15, codec="libx264", macro_block_size=16)
    total = len(records)
    for r in records:
        frame = _compose_frame(
            r["rgb_cam"],
            r["xyz"],
            r["rgb_pc"],
            r["ee"],
            r["ach"],
            r["action"],
            task_id,
            mode_label,
            r["step"],
            total,
            r["disp"],
            r["success"],
        )
        # Pad width/height to multiple of 16 for clean H.264 encoding
        fh, fw = frame.shape[:2]
        pw = ((fw + 15) // 16) * 16 - fw
        ph = ((fh + 15) // 16) * 16 - fh
        if pw or ph:
            frame = cv2.copyMakeBorder(frame, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(18, 18, 18))
        writer.append_data(frame)
    writer.close()

    # 6 filmstrip thumbnails (RGB + PC side-by-side)
    idxs = np.linspace(0, max(0, total - 1), num=6, dtype=int)
    thumbs = []
    for idx in idxs:
        r = records[int(idx)]
        cam_s = cv2.resize(r["rgb_cam"], (180, 180), interpolation=cv2.INTER_AREA)
        pc_s = cv2.resize(_draw_pointcloud_panel(r["xyz"], r["rgb_pc"], r["ee"], r["ach"], DEFAULT_BOUNDS, size=224), (180, 180), interpolation=cv2.INTER_AREA)
        pair = np.concatenate([cam_s, pc_s], axis=1)
        cv2.putText(pair, f"t={r['step']}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(pair)

    # Compact 16-frame quantized widget payload (120 points/frame)
    w_idxs = np.linspace(0, max(0, total - 1), num=min(16, total), dtype=int)
    rng = np.random.default_rng(0)
    widget_frames: list[dict[str, Any]] = []
    for idx in w_idxs:
        r = records[int(idx)]
        sel = rng.choice(r["xyz"].shape[0], size=120, replace=False)
        pts = np.round(r["xyz"][sel], 3).tolist()
        cols = r["rgb_pc"][sel].tolist()
        widget_frames.append(
            {
                "t": int(r["step"]),
                "pts": pts,
                "rgb": cols,
                "ee": np.round(r["ee"], 3).tolist(),
                "ach": np.round(r["ach"], 3).tolist(),
                "act": np.round(r["action"], 2).tolist(),
                "disp_cm": round(r["disp"] * 100.0, 1),
                "succ": bool(r["success"]),
            }
        )
    return thumbs, widget_frames


def _build_filmstrip_png(rows: list[tuple[str, list[np.ndarray]]], out_png: Path, title: str) -> None:
    fig, axes = plt.subplots(len(rows), 6, figsize=(21, 3.1 * len(rows)), dpi=110)
    fig.patch.set_facecolor("#111318")
    for r_idx, (label, thumbs) in enumerate(rows):
        for c_idx, img in enumerate(thumbs):
            ax = axes[r_idx, c_idx]
            ax.imshow(img)
            ax.axis("off")
            if c_idx == 0:
                ax.set_title(label, color="#e5e7eb", fontsize=10, loc="left", pad=4)
    fig.suptitle(title, color="#f9fafb", fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_and_pace(env: Any, dt: float) -> bool:
    """Render one frame in ManiSkill3 SAPIEN GUI, pace execution, return True if window closed."""
    viewer = env.render()
    if dt > 0:
        time.sleep(dt)
    return bool(getattr(viewer, "closed", False))


def _pause_between_modes(env: Any, seconds: float = 0.8) -> bool:
    """Keep SAPIEN GUI responsive between episodes so camera interaction stays smooth."""
    tick = 0.05
    steps = max(1, int(round(seconds / tick)))
    for _ in range(steps):
        if _render_and_pace(env, tick):
            return True
    return False


def _play_human_expert(
    env: Any,
    expert: SRMPExpertController,
    task_id: str,
    max_steps: int,
    seed: int,
    dt: float,
    render_live: bool,
) -> tuple[dict[str, dict[str, Any]], int, float, bool, bool]:
    """Run SRMP expert rollout to capture stage snapshots and optionally render live in GUI."""
    device = env.unwrapped.device
    stages = None
    for attempt in range(10):
        env.reset(seed=seed + attempt)
        stages = expert.plan_episode()
        if stages is not None:
            break
    if stages is None:
        raise RuntimeError(f"SRMP failed across 10 seeds for {task_id}")

    snaps: dict[str, dict[str, Any]] = {}
    key_stages = {"approach", "grasp", "lift", "transport", "place_or_pull"}
    ach0 = None
    step_idx = 0
    max_disp = 0.0
    any_succ = False
    closed = False

    if render_live:
        print(f"\n=== [{task_id}] Expert SRMP ===", flush=True)
        if _render_and_pace(env, dt):
            return snaps, step_idx, max_disp, any_succ, True

    for s_idx, stage in enumerate(stages):
        if stage.name in key_stages:
            rem_wps, rem_grips = _flatten_stages(stages[s_idx:])
            snaps[stage.name] = {
                "stage": stage.name,
                "is_grasped": bool(stage.is_grasped),
                "state_dict": _squeeze_state_dict_to_cpu(env.unwrapped.get_state_dict()),
                "remaining_waypoints": rem_wps,
                "remaining_grippers": rem_grips,
            }

        targets = list(stage.waypoints) + [stage.waypoints[-1]] * stage.settle_steps

        def _step_to(wp_target: np.ndarray) -> bool:
            nonlocal step_idx, ach0, max_disp, any_succ, closed
            if step_idx >= max_steps or closed:
                return True
            q_curr = env.unwrapped.agent.robot.get_qpos()[0, :7].cpu().numpy()
            act_np = joint_target_to_delta_action(q_curr, wp_target, stage.gripper, ARM_DELTA_SCALE)
            act_t = torch.from_numpy(act_np).unsqueeze(0).to(device)
            meta = extract_step_metadata(env, task_id)
            ach = meta["achieved_point"][0].cpu().numpy()
            if ach0 is None:
                ach0 = ach.copy()
            _, _, term, trunc, _ = env.step(act_t)
            ev = env.unwrapped.evaluate()
            succ = bool(ev.get("success", torch.tensor([False]))[0])
            disp = float(np.linalg.norm(ach - ach0))
            max_disp = max(max_disp, disp)
            any_succ = any_succ or succ
            step_idx += 1
            if render_live and _render_and_pace(env, dt):
                closed = True
                return True
            return bool(term[0] or trunc[0])

        done = False
        max_wp_tries = 3 if stage.name == "place_or_pull" else 1
        for wp in targets:
            for _ in range(max_wp_tries):
                done = _step_to(wp)
                if done:
                    break
                q_curr = env.unwrapped.agent.robot.get_qpos()[0, :7].cpu().numpy()
                if float(np.max(np.abs(q_curr - wp))) <= 0.035:
                    break
            if done:
                break
        if not done and stage.name in {"grasp", "lower_hook", "place_or_pull"}:
            final_wp = stage.waypoints[-1]
            for _ in range(8):
                q_curr = env.unwrapped.agent.robot.get_qpos()[0, :7].cpu().numpy()
                if float(np.max(np.abs(q_curr - final_wp))) < 0.015:
                    break
                done = _step_to(final_wp)
                if done:
                    break
        if done or step_idx >= max_steps or closed:
            break

    if render_live:
        print(
            f"[{task_id}] {'expert_srmp':24s} steps={step_idx:3d} max_disp={max_disp * 100.0:5.1f}cm success={any_succ}",
            flush=True,
        )
    return snaps, step_idx, max_disp * 100.0, any_succ, closed


def _play_human_exploratory(
    env: Any,
    task_id: str,
    mode_key: str,
    mode_label: str,
    snaps: dict[str, dict[str, Any]],
    max_steps: int,
    seed: int,
    dt: float,
) -> bool:
    """Replay one snapshot-branched or random exploratory mode live in SAPIEN GUI."""
    device = env.unwrapped.device
    gen = torch.Generator(device=device).manual_seed(seed)
    rem_wps, rem_grips, rollout_steps = _prepare_exploratory_episode(env, mode_key, snaps, max_steps, seed)

    ou_state = torch.zeros((1, 7), device=device, dtype=torch.float32)
    t_split = max(10, rollout_steps // 3)
    ach0 = None
    max_disp = 0.0
    any_succ = False

    print(f"\n=== [{task_id}] {mode_label} ===", flush=True)
    if _render_and_pace(env, dt):
        return True

    for t in range(rollout_steps):
        meta = extract_step_metadata(env, task_id)
        ach = meta["achieved_point"][0].cpu().numpy()
        if ach0 is None:
            ach0 = ach.copy()

        q_curr = env.unwrapped.agent.robot.get_qpos()[:, :7]
        idx = min(t, len(rem_wps) - 1)
        exp_act = joint_target_to_delta_action(
            q_curr, torch.from_numpy(rem_wps[idx : idx + 1]).to(device), float(rem_grips[idx]), ARM_DELTA_SCALE
        )
        ou_state = _sample_ou_noise(ou_state, 0.15, 0.30, gen)
        act = _compute_exploratory_action(mode_key, t, t_split, exp_act, ou_state, gen, device)

        env.step(act)
        ev = env.unwrapped.evaluate()
        succ = bool(ev.get("success", torch.tensor([False]))[0])
        disp = float(np.linalg.norm(ach - ach0))
        max_disp = max(max_disp, disp)
        any_succ = any_succ or succ
        if _render_and_pace(env, dt):
            return True

    print(
        f"[{task_id}] {mode_key:24s} steps={rollout_steps:3d} max_disp={max_disp * 100.0:5.1f}cm success={any_succ}",
        flush=True,
    )
    return False


def _run_human_viewer(args: argparse.Namespace) -> None:
    """Open native ManiSkill3 SAPIEN GUI (render_mode='human') and step rollouts live."""
    label_map = dict(MODE_LABELS)
    dt = 1.0 / max(1.0, float(args.fps))
    requested_modes = list(args.modes)

    cycle = 0
    try:
        while True:
            for t_idx, task_id in enumerate(args.tasks):
                env = gym.make(
                    task_id,
                    num_envs=1,
                    obs_mode="none",
                    control_mode="pd_joint_delta_pos",
                    sim_backend="physx_cuda",
                    render_mode="human",
                    max_episode_steps=args.max_steps,
                )
                try:
                    expert = SRMPExpertController(env, task_id)
                    exp_seed = args.seed + cycle * 1000 + t_idx * 100
                    # Always collect SRMP stage snapshots; render live if expert_srmp is first mode.
                    play_exp_now = len(requested_modes) > 0 and requested_modes[0] == "expert_srmp"
                    snaps, _, _, _, closed = _play_human_expert(
                        env, expert, task_id, args.max_steps, exp_seed, dt, render_live=play_exp_now
                    )
                    if closed:
                        return
                    if play_exp_now and len(requested_modes) > 1 and _pause_between_modes(env, 0.8):
                        return

                    start_idx = 1 if play_exp_now else 0
                    for m_offset, mode_key in enumerate(requested_modes[start_idx:]):
                        m_idx = list(label_map.keys()).index(mode_key)
                        if mode_key == "expert_srmp":
                            _, _, _, _, closed = _play_human_expert(
                                env, expert, task_id, args.max_steps, exp_seed, dt, render_live=True
                            )
                            if closed:
                                return
                        else:
                            closed = _play_human_exploratory(
                                env,
                                task_id,
                                mode_key,
                                label_map[mode_key],
                                snaps,
                                args.max_steps,
                                args.seed + cycle * 1000 + 500 + m_idx,
                                dt,
                            )
                            if closed:
                                return
                        is_last_mode = (start_idx + m_offset) == (len(requested_modes) - 1)
                        if not is_last_mode and _pause_between_modes(env, 0.8):
                            return
                finally:
                    env.close()

            cycle += 1
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\nInterrupted live ManiSkill3 viewer.", flush=True)


def main() -> None:
    all_mode_keys = [k for k, _ in MODE_LABELS]
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["StackCube-v1", "PullCubeTool-v1"])
    parser.add_argument("--modes", nargs="+", choices=all_mode_keys, default=all_mode_keys)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/visual_verify"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("outputs/visual_verify"))
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--human", action="store_true", help="Open interactive ManiSkill3 SAPIEN GUI viewer.")
    parser.add_argument("--fps", type=float, default=20.0, help="Playback FPS for --human mode.")
    parser.add_argument("--loop", action="store_true", help="Loop tasks/modes continuously in --human mode.")
    parser.add_argument("--save-zarr-dir", type=Path, default=None, help="Save 6D point-cloud trajectories per task as Zarr v3.")
    parser.add_argument("--collect-zarr-only", action="store_true", help="Skip MP4/filmstrip/widget outputs and only collect Zarr v3.")
    args = parser.parse_args()

    if args.human:
        _run_human_viewer(args)
        return

    render_rgb = not args.collect_zarr_only
    widget_payload: dict[str, Any] = {"bounds": DEFAULT_BOUNDS, "tasks": {}}

    for t_idx, task_id in enumerate(args.tasks):
        env = gym.make(
            task_id,
            num_envs=1,
            obs_mode="pointcloud",
            control_mode="pd_joint_delta_pos",
            sim_backend="physx_cuda",
            render_mode="rgb_array" if render_rgb else None,
            max_episode_steps=args.max_steps,
        )
        expert = SRMPExpertController(env, task_id)
        exp_records, snaps = _rollout_expert(env, expert, task_id, args.max_steps, args.seed + t_idx * 100, render_rgb=render_rgb)

        writer: ZarrTrajectoryWriter | None = None
        if args.save_zarr_dir is not None:
            zarr_path = args.save_zarr_dir / f"{task_id}_modes.zarr"
            writer = ZarrTrajectoryWriter(
                zarr_path,
                num_points=4096,
                action_dim=8,
                proprio_dim=26,
                attrs={"task_id": task_id, "bounds": [list(b) for b in DEFAULT_BOUNDS]},
            )
            ep_modes: list[str] = []
            ep_labels: list[str] = []
            ep_success: list[bool] = []

        rows: list[tuple[str, list[np.ndarray]]] = []
        task_modes: dict[str, Any] = {}

        for m_idx, (mode_key, mode_label) in enumerate(MODE_LABELS):
            if mode_key == "expert_srmp":
                recs = exp_records
            else:
                recs = _rollout_exploratory(
                    env, task_id, mode_key, snaps, args.max_steps, args.seed + 500 + m_idx, render_rgb=render_rgb
                )

            max_disp = max(r["disp"] for r in recs) * 100.0
            any_succ = any(r["success"] for r in recs)

            if writer is not None:
                writer.append_episode(
                    xyz=np.stack([r["xyz"] for r in recs], axis=0),
                    rgb=np.stack([r["rgb_pc"] for r in recs], axis=0),
                    action=np.stack([r["action"] for r in recs], axis=0),
                    proprio=np.stack([r["proprio"] for r in recs], axis=0),
                    achieved_point=np.stack([r["ach"] for r in recs], axis=0),
                    ee_point=np.stack([r["ee"] for r in recs], axis=0),
                )
                ep_modes.append(mode_key)
                ep_labels.append(mode_label)
                ep_success.append(bool(any_succ))

            if render_rgb:
                mp4_path = args.out_dir / f"{task_id}_{mode_key}.mp4"
                thumbs, w_frames = _save_video_and_filmstrip_row(recs, task_id, mode_key, mode_label, mp4_path)
                rows.append((mode_label, thumbs))
                task_modes[mode_key] = {"label": mode_label, "frames": w_frames, "max_disp_cm": round(max_disp, 1), "success": any_succ}
                print(f"[{task_id}] {mode_key:24s} steps={len(recs):3d} max_disp={max_disp:5.1f}cm success={any_succ} -> {mp4_path}")
            else:
                print(f"[{task_id}] {mode_key:24s} steps={len(recs):3d} max_disp={max_disp:5.1f}cm success={any_succ}")

        env.close()
        if writer is not None:
            writer.root.attrs["episode_modes"] = ep_modes
            writer.root.attrs["episode_labels"] = ep_labels
            writer.root.attrs["episode_success"] = ep_success
            print(f"[{task_id}] Saved Zarr v3 ({writer.num_episodes} episodes, {writer.total_steps} steps) -> {writer.path}")

        if render_rgb:
            short = "stackcube" if "Stack" in task_id else "pullcube"
            png_path = args.artifact_dir / f"filmstrip_{short}.png"
            _build_filmstrip_png(rows, png_path, f"{task_id} — 6 Trajectory Modes (RGB Camera + Preprocessed 6D Point Cloud)")
            widget_payload["tasks"][task_id] = task_modes

    if render_rgb:
        scratch_json = args.artifact_dir / "scratch" / "rollout_widget_data.json"
        scratch_json.parent.mkdir(parents=True, exist_ok=True)
        scratch_json.write_text(json.dumps(widget_payload))
        print("Saved widget data:", scratch_json)


if __name__ == "__main__":
    main()
