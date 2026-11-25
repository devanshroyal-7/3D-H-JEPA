"""Interactive 3D point-cloud trajectory stepper built with Viser.

Loads Zarr v3 trajectories produced by `visualize_rollouts.py --save-zarr-dir`
or `collect_maniskill3_6d.py`, letting the user switch between tasks/episodes,
step frame-by-frame (`t = 0 .. T-1`), scrub/play, and toggle a previous-frame
`(t-1)` ghost overlay plus EE/object markers and trails.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import viser
import zarr

from point_hjepa.data.collect_maniskill3_6d import DEFAULT_BOUNDS


@dataclass
class EpisodeData:
    name: str
    task_id: str
    mode_label: str
    xyz: np.ndarray            # (T, N, 3) float32
    rgb: np.ndarray            # (T, N, 3) uint8
    action: np.ndarray         # (T, 8) float32
    ee_point: np.ndarray       # (T, 3) float32
    achieved_point: np.ndarray # (T, 3) float32
    disp_cm: np.ndarray        # (T,) float32
    success_ep: bool | None


def _bounds_segments(bounds: list[list[float]] | tuple[tuple[float, float], ...]) -> tuple[np.ndarray, np.ndarray]:
    (x0, x1), (y0, y1), (z0, z1) = bounds
    corners = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    edge_idx = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    pts = np.stack([[corners[a], corners[b]] for a, b in edge_idx], axis=0).astype(np.float32)
    cols = np.full((len(edge_idx), 2, 3), 130, dtype=np.uint8)
    return pts, cols


def _path_segments(pts: np.ndarray, color_rgb: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(T-1, 2, 3)`` line segments from ``(T, 3)`` polyline points."""
    segs = np.stack([pts[:-1], pts[1:]], axis=1).astype(np.float32)
    cols = np.broadcast_to(np.array(color_rgb, dtype=np.uint8), segs.shape).copy()
    return segs, cols


def _load_zarr_episodes(paths: list[Path]) -> tuple[list[EpisodeData], tuple[tuple[float, float], ...]]:
    episodes: list[EpisodeData] = []
    bounds: Any = DEFAULT_BOUNDS

    for zpath in paths:
        if not zpath.exists():
            raise FileNotFoundError(f"Zarr store not found: {zpath}")
        root = zarr.open_group(str(zpath), mode="r")
        attrs = dict(root.attrs)
        task_id = str(attrs.get("task_id", zpath.stem))
        if "bounds" in attrs:
            bounds = attrs["bounds"]

        xyz_all = np.asarray(root["xyz"][:], dtype=np.float32)
        rgb_all = np.asarray(root["rgb"][:], dtype=np.uint8)
        act_all = np.asarray(root["action"][:], dtype=np.float32)
        ee_all = np.asarray(root["ee_point"][:], dtype=np.float32)
        ach_all = np.asarray(root["achieved_point"][:], dtype=np.float32)
        ends = np.asarray(root["episode_ends"][:], dtype=np.int64)

        ep_labels = attrs.get("episode_labels")
        ep_modes = attrs.get("episode_modes")
        ep_success = attrs.get("episode_success")

        start = 0
        for i, end in enumerate(ends.tolist()):
            end_i = int(end)
            if end_i <= start:
                start = end_i
                continue
            ach_ep = ach_all[start:end_i]
            disp_cm = np.linalg.norm(ach_ep - ach_ep[0:1], axis=-1) * 100.0
            T = end_i - start

            if isinstance(ep_labels, list) and i < len(ep_labels):
                mode_label = str(ep_labels[i])
            elif isinstance(ep_modes, list) and i < len(ep_modes):
                mode_label = str(ep_modes[i])
            else:
                mode_label = f"ep {i:03d}"

            succ: bool | None = None
            if isinstance(ep_success, list) and i < len(ep_success):
                succ = bool(ep_success[i])

            tag = " [PASS]" if succ is True else (" [FAIL]" if succ is False else "")
            disp_name = f"{task_id} | {mode_label} ({T} steps, max {float(disp_cm.max()):.1f}cm){tag}"
            # Disambiguate if duplicate names exist across stores
            existing = {e.name for e in episodes}
            if disp_name in existing:
                disp_name = f"{zpath.stem} #{i} | {mode_label} ({T} steps)"

            episodes.append(
                EpisodeData(
                    name=disp_name,
                    task_id=task_id,
                    mode_label=mode_label,
                    xyz=xyz_all[start:end_i],
                    rgb=rgb_all[start:end_i],
                    action=act_all[start:end_i],
                    ee_point=ee_all[start:end_i],
                    achieved_point=ach_ep,
                    disp_cm=disp_cm,
                    success_ep=succ,
                )
            )
            start = end_i

    return episodes, bounds


def _format_telemetry(ep: EpisodeData, t: int) -> str:
    act = ep.action[t]
    ee = ep.ee_point[t]
    ach = ep.achieved_point[t]
    t_max = ep.xyz.shape[0] - 1
    succ_txt = "PASS" if ep.success_ep else ("NO-GOAL" if ep.success_ep is False else "N/A")
    lines = [
        f"**Task:** `{ep.task_id}`  ",
        f"**Mode:** `{ep.mode_label}` ({succ_txt})  ",
        f"**Step:** `t = {t} / {t_max}` | **Cube disp:** `{ep.disp_cm[t]:.1f} cm` (max `{ep.disp_cm.max():.1f} cm`)  ",
        f"**EE (cyan):** `({ee[0]:+.3f}, {ee[1]:+.3f}, {ee[2]:+.3f})` m  ",
        f"**Obj (amber):** `({ach[0]:+.3f}, {ach[1]:+.3f}, {ach[2]:+.3f})` m  ",
        f"**Action (8D):** `[{', '.join(f'{v:+.2f}' for v in act[:7])}, grp={act[7]:+.1f}]`",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarrs",
        nargs="*",
        type=Path,
        default=None,
        help="One or more Zarr v3 trajectory stores (default: outputs/pc_inspect/*_modes.zarr).",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    zarr_paths = list(args.zarrs) if args.zarrs else sorted(Path("outputs/pc_inspect").glob("*_modes.zarr"))
    if not zarr_paths:
        print(
            "No Zarr stores found under outputs/pc_inspect/*_modes.zarr.\n"
            "Collect first with:\n"
            "  .venv/bin/python scripts/visualize_rollouts.py --save-zarr-dir outputs/pc_inspect --collect-zarr-only",
            file=sys.stderr,
        )
        raise SystemExit(1)

    episodes, bounds = _load_zarr_episodes(zarr_paths)
    if not episodes:
        raise RuntimeError(f"No episodes found in: {zarr_paths}")
    ep_map = {ep.name: ep for ep in episodes}

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (1.25, -0.75, 0.72)
        client.camera.look_at = (0.45, 0.0, 0.16)

    # Static table grid & workspace bounds
    server.scene.add_grid(
        "/grid",
        width=2.0,
        height=2.0,
        plane="xy",
        cell_size=0.1,
        section_size=0.5,
        position=(0.45, 0.0, 0.0),
    )
    b_pts, b_cols = _bounds_segments(bounds)
    bounds_handle = server.scene.add_line_segments(
        "/bounds",
        points=b_pts,
        colors=b_cols,
        thickness=1.5,
        thickness_units="screen",
    )

    ep0 = episodes[0]
    cloud_curr = server.scene.add_point_cloud(
        "/cloud/current",
        points=ep0.xyz[0],
        colors=ep0.rgb[0],
        point_size=0.009,
        point_shape="circle",
        precision="float32",
    )
    ghost_cols = np.full(ep0.xyz[0].shape, (255, 80, 140), dtype=np.uint8)
    cloud_prev = server.scene.add_point_cloud(
        "/cloud/prev_ghost",
        points=ep0.xyz[0],
        colors=ghost_cols,
        point_size=0.007,
        point_shape="circle",
        precision="float32",
        visible=False,
    )
    ee_marker = server.scene.add_icosphere("/markers/ee", radius=0.015, color=(6, 182, 212), position=ep0.ee_point[0])
    ach_marker = server.scene.add_icosphere("/markers/achieved", radius=0.018, color=(245, 158, 11), position=ep0.achieved_point[0])
    dummy_seg = np.zeros((1, 2, 3), dtype=np.float32)
    ee_trail = server.scene.add_line_segments("/trails/ee", points=dummy_seg, colors=np.full((1, 2, 3), (6, 182, 212), dtype=np.uint8), thickness=2.0, thickness_units="screen", visible=False)
    ach_trail = server.scene.add_line_segments("/trails/achieved", points=dummy_seg, colors=np.full((1, 2, 3), (245, 158, 11), dtype=np.uint8), thickness=2.0, thickness_units="screen", visible=False)

    # GUI controls
    ep_dropdown = server.gui.add_dropdown("Episode", options=[ep.name for ep in episodes], initial_value=ep0.name)
    step_slider = server.gui.add_slider("Step t", min=0, max=max(1, ep0.xyz.shape[0] - 1), step=1, initial_value=0)
    prev_btn = server.gui.add_button("Prev (-1)")
    next_btn = server.gui.add_button("Next (+1)")
    back5_btn = server.gui.add_button("Back (-5)")
    fwd5_btn = server.gui.add_button("Fwd (+5)")
    reset_btn = server.gui.add_button("Reset (t=0)")
    play_cb = server.gui.add_checkbox("Play", initial_value=False)
    fps_slider = server.gui.add_slider("FPS", min=1, max=30, step=1, initial_value=10)
    pt_size_slider = server.gui.add_slider("Point Size (m)", min=0.003, max=0.025, step=0.001, initial_value=0.009)
    ghost_cb = server.gui.add_checkbox("Show Previous Frame (t-1)", initial_value=False)
    markers_cb = server.gui.add_checkbox("Show EE / Object + Trails", initial_value=True)
    bounds_cb = server.gui.add_checkbox("Show Workspace Bounds", initial_value=True)
    telemetry_md = server.gui.add_markdown(_format_telemetry(ep0, 0))

    lock = threading.RLock()
    state = {"ep": ep0}

    def render_frame() -> None:
        with lock:
            ep: EpisodeData = state["ep"]
            t_max = ep.xyz.shape[0] - 1
            t = int(np.clip(step_slider.value, 0, t_max))

            cloud_curr.points = ep.xyz[t]
            cloud_curr.colors = ep.rgb[t]
            cloud_curr.point_size = float(pt_size_slider.value)

            if ghost_cb.value and t > 0:
                cloud_prev.points = ep.xyz[t - 1]
                cloud_prev.colors = np.full(ep.xyz[t - 1].shape, (255, 80, 140), dtype=np.uint8)
                cloud_prev.point_size = max(0.003, float(pt_size_slider.value) * 0.8)
                cloud_prev.visible = True
            else:
                cloud_prev.visible = False

            show_m = bool(markers_cb.value)
            ee_marker.position = ep.ee_point[t]
            ee_marker.visible = show_m
            ach_marker.position = ep.achieved_point[t]
            ach_marker.visible = show_m

            if show_m and t >= 1:
                ee_segs, ee_cols = _path_segments(ep.ee_point[: t + 1], (6, 182, 212))
                ee_trail.points = ee_segs
                ee_trail.colors = ee_cols
                ee_trail.visible = True

                ach_segs, ach_cols = _path_segments(ep.achieved_point[: t + 1], (245, 158, 11))
                ach_trail.points = ach_segs
                ach_trail.colors = ach_cols
                ach_trail.visible = True
            else:
                ee_trail.visible = False
                ach_trail.visible = False

            bounds_handle.visible = bool(bounds_cb.value)
            telemetry_md.content = _format_telemetry(ep, t)

    @ep_dropdown.on_update
    def _(_: Any) -> None:
        with lock:
            ep = ep_map[ep_dropdown.value]
            state["ep"] = ep
            step_slider.max = max(1, ep.xyz.shape[0] - 1)
            step_slider.value = 0
            render_frame()

    @step_slider.on_update
    def _(_: Any) -> None:
        render_frame()

    @pt_size_slider.on_update
    def _(_: Any) -> None:
        render_frame()

    @ghost_cb.on_update
    def _(_: Any) -> None:
        render_frame()

    @markers_cb.on_update
    def _(_: Any) -> None:
        render_frame()

    @bounds_cb.on_update
    def _(_: Any) -> None:
        render_frame()

    @prev_btn.on_click
    def _(_: Any) -> None:
        with lock:
            play_cb.value = False
            step_slider.value = max(0, int(step_slider.value) - 1)
            render_frame()

    @next_btn.on_click
    def _(_: Any) -> None:
        with lock:
            play_cb.value = False
            t_max = state["ep"].xyz.shape[0] - 1
            step_slider.value = min(t_max, int(step_slider.value) + 1)
            render_frame()

    @back5_btn.on_click
    def _(_: Any) -> None:
        with lock:
            play_cb.value = False
            step_slider.value = max(0, int(step_slider.value) - 5)
            render_frame()

    @fwd5_btn.on_click
    def _(_: Any) -> None:
        with lock:
            play_cb.value = False
            t_max = state["ep"].xyz.shape[0] - 1
            step_slider.value = min(t_max, int(step_slider.value) + 5)
            render_frame()

    @reset_btn.on_click
    def _(_: Any) -> None:
        with lock:
            play_cb.value = False
            step_slider.value = 0
            render_frame()

    render_frame()
    print(f"Viser 3D Point-Cloud Stepper listening at http://{args.host}:{args.port} ({len(episodes)} episodes loaded)", flush=True)

    try:
        while True:
            fps = max(1, int(fps_slider.value))
            time.sleep(1.0 / fps)
            if play_cb.value:
                with lock:
                    t_max = state["ep"].xyz.shape[0] - 1
                    nxt = int(step_slider.value) + 1
                    step_slider.value = 0 if nxt > t_max else nxt
                    render_frame()
    except KeyboardInterrupt:
        print("\nStopping Viser server.", flush=True)
        server.stop()


if __name__ == "__main__":
    main()
