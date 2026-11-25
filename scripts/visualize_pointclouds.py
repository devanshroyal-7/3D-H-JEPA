"""Render collected point clouds with the course PyTorch3D point renderer.

The orbit GIF is the same setup as ``get_points_renderer`` / ``viz_seg`` in the
learning-3D assignments: ``PointsRasterizer`` + ``AlphaCompositor``, a ring of
``look_at_view_transform`` cameras, and ``imageio.mimsave``. Robot clouds are
Z-up; the renderer is Y-up, so points are rotated before they go in.

PyTorch3D is not a dependency of this repo (it is installed in the learning3d
conda env). Example:

    /home/devansh/miniconda3/envs/learning3d/bin/python scripts/visualize_pointclouds.py \\
        outputs/explore/pickcube.zarr --out outputs/explore/previews
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SOURCE_NAMES = ("expert", "noisy", "explore")
# Meters per scene unit. The course cameras sit ~3 units from the origin and
# expect a cloud of about that size, so a tabletop stays in frame.
SCENE_METERS = 0.45


def get_points_renderer(image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)):
    """PyTorch3D point-cloud renderer, matching the learning-3D course helper."""
    import torch
    from pytorch3d.renderer import (
        AlphaCompositor,
        PointsRasterizationSettings,
        PointsRasterizer,
        PointsRenderer,
    )

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius)
    return PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )


def _open(path: Path):
    if path.suffix == ".npz":
        return np.load(path), None
    if path.suffix in {".h5", ".hdf5"} or path.is_file():
        import h5py

        handle = h5py.File(path, "r")
        return handle, handle.close
    import zarr

    return zarr.open_group(path, mode="r"), None


def export_npz(dataset: Path, destination: Path) -> Path:
    """Copy the arrays the renderer needs into one file the course env can open."""
    root, close = _open(dataset)
    try:
        payload = {
            "points": np.asarray(root["points"][:]),
            "point_count": np.asarray(root["point_count"][:]),
            "goal_coords": np.asarray(root["goal_coords"][:]),
            "robot_state": np.asarray(root["robot_state"][:]),
            "episode_ends": np.asarray(root["episode_ends"][:]),
            "episode_source": np.asarray(root["episode_source"][:]),
            "episode_success": np.asarray(root["episode_success"][:]),
        }
        if "rgb" in root:
            payload["rgb"] = np.asarray(root["rgb"][:])
    finally:
        if close is not None:
            close()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return destination


def _episode_spans(ends: np.ndarray) -> list[tuple[int, int]]:
    start = 0
    spans = []
    for end in ends.tolist():
        spans.append((start, int(end)))
        start = int(end)
    return spans


def z_up_to_y_up(points: np.ndarray) -> np.ndarray:
    """Robot base frame (Z up) to the Y-up frame the course cameras assume."""
    out = np.empty(points.shape, dtype=np.float32)
    out[..., 0] = points[..., 0]
    out[..., 1] = points[..., 2]
    out[..., 2] = -points[..., 1]
    return out


def _valid(points: np.ndarray, count: int) -> np.ndarray:
    count = int(np.clip(count, 0, points.shape[0]))
    cloud = points[:count]
    finite = np.isfinite(cloud).all(axis=1)
    return cloud[finite]


def _height_colors(points: np.ndarray) -> np.ndarray:
    z = points[:, 2]
    span = float(z.max() - z.min()) if len(z) else 1.0
    t = (z - z.min()) / max(span, 1e-6)
    # A short blue-teal-amber ramp, so the table and the arm separate without RGB.
    stops = np.array(
        [
            [0.15, 0.25, 0.55],
            [0.15, 0.65, 0.65],
            [0.95, 0.75, 0.20],
            [0.85, 0.25, 0.20],
        ],
        dtype=np.float32,
    )
    idx = t * (len(stops) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    frac = (idx - lo)[:, None]
    return stops[lo] * (1 - frac) + stops[hi] * frac


def _marker(center: np.ndarray, color: np.ndarray, radius: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=np.float32,
    )
    points = center.reshape(1, 3) + offsets * radius
    colors = np.repeat(color.reshape(1, 3), len(points), axis=0)
    return points, colors


def _frame_cloud(root, frame: int, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = int(root["point_count"][frame])
    raw = np.asarray(root["points"][frame])[:count]
    finite = np.isfinite(raw).all(axis=1)
    points = raw[finite]
    if "rgb" in root:
        colors = np.asarray(root["rgb"][frame])[:count][finite].astype(np.float32) / 255.0
    else:
        colors = _height_colors(points)
    extras_p = []
    extras_c = []
    goal = np.asarray(root["goal_coords"][frame], dtype=np.float32).reshape(-1)
    if goal.size >= 3 and np.isfinite(goal[:3]).all():
        goal_pts, goal_cols = _marker(goal[:3], np.array([1.0, 0.1, 0.1], dtype=np.float32))
        extras_p.append(goal_pts)
        extras_c.append(goal_cols)
    state = np.asarray(root["robot_state"][frame], dtype=np.float32).reshape(-1)
    if state.size >= 21 and np.isfinite(state[18:21]).all():
        tcp_pts, tcp_cols = _marker(state[18:21], np.array([0.1, 0.85, 0.2], dtype=np.float32), 0.015)
        extras_p.append(tcp_pts)
        extras_c.append(tcp_cols)
    if extras_p:
        points = np.concatenate([points, *extras_p], axis=0)
        colors = np.concatenate([colors, *extras_c], axis=0)
    shifted = z_up_to_y_up(points - center) / SCENE_METERS
    return shifted.astype(np.float32), np.clip(colors, 0, 1).astype(np.float32)


def _render(points: np.ndarray, colors: np.ndarray, azim, elev, dist, image_size, radius, device):
    import pytorch3d
    import torch
    from pytorch3d.renderer import FoVPerspectiveCameras

    n = len(azim)
    verts = torch.as_tensor(points, device=device).unsqueeze(0).repeat(n, 1, 1)
    feats = torch.as_tensor(colors, device=device).unsqueeze(0).repeat(n, 1, 1)
    cloud = pytorch3d.structures.Pointclouds(points=verts, features=feats)
    rotation, translation = pytorch3d.renderer.cameras.look_at_view_transform(
        dist=dist, elev=elev, azim=azim, device=device
    )
    cameras = FoVPerspectiveCameras(R=rotation, T=translation, fov=60, device=device)
    renderer = get_points_renderer(
        image_size=image_size, device=device, radius=radius, background_color=(1, 1, 1)
    )
    images = renderer(cloud, cameras=cameras).detach().cpu().numpy()
    return (np.clip(images, 0, 1) * 255).astype(np.uint8)


def _caption(frame: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    bar = Image.new("RGB", (image.width, 22), (255, 255, 255))
    draw = ImageDraw.Draw(bar)
    draw.text((6, 4), text, fill=(20, 24, 32))
    canvas = Image.new("RGB", (image.width, image.height + 22), (255, 255, 255))
    canvas.paste(bar, (0, 0))
    canvas.paste(image, (0, 22))
    return np.asarray(canvas)


def render_dataset(path: Path, out: Path, image_size: int, views: int, motion_stride: int) -> None:
    import imageio
    import torch

    root, close = _open(path)
    try:
        ends = np.asarray(root["episode_ends"][:], dtype=np.int64)
        sources = np.asarray(root["episode_source"][:], dtype=np.int8)
        success = np.asarray(root["episode_success"][:], dtype=bool)
        spans = _episode_spans(ends)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        out.mkdir(parents=True, exist_ok=True)
        sheet = []
        azim = [180 - (360 / views) * i for i in range(views)]
        for episode, ((start, end), source, ok) in enumerate(zip(spans, sources, success)):
            name = SOURCE_NAMES[int(source)] if 0 <= int(source) < len(SOURCE_NAMES) else str(int(source))
            tag = f"ep{episode:02d}_{name}_{'success' if ok else 'fail'}"
            count0 = int(root["point_count"][start])
            seed_points = _valid(np.asarray(root["points"][start]), count0)
            center = np.median(seed_points, axis=0).astype(np.float32)
            keyframes = [start, (start + end - 1) // 2, end - 1]
            for frame in keyframes:
                local = frame - start
                pts, cols = _frame_cloud(root, frame, center)
                rendered = _render(
                    pts, cols, azim=azim, elev=18, dist=3.2, image_size=image_size, radius=0.012, device=device
                )
                caption = f"{tag}  t={local}  orbit  red=goal green=tcp"
                frames = [_caption(image, caption) for image in rendered]
                gif = out / f"{tag}_orbit_t{local:03d}.gif"
                imageio.mimsave(gif, frames, fps=12, loop=0)
                sheet.append(frames[views // 3])
                print(f"wrote {gif}")
            motion_idx = list(range(start, end, motion_stride))
            if motion_idx[-1] != end - 1:
                motion_idx.append(end - 1)
            motion = []
            for frame in motion_idx:
                pts, cols = _frame_cloud(root, frame, center)
                image = _render(
                    pts,
                    cols,
                    azim=[140],
                    elev=22,
                    dist=3.2,
                    image_size=image_size,
                    radius=0.012,
                    device=device,
                )[0]
                motion.append(_caption(image, f"{tag}  t={frame - start}/{end - start - 1}  fixed camera"))
            gif = out / f"{tag}_motion.gif"
            imageio.mimsave(gif, motion, fps=6, loop=0)
            print(f"wrote {gif}")
        if sheet:
            tile = sheet[0].shape[0]
            cols = min(3, len(sheet))
            rows = int(np.ceil(len(sheet) / cols))
            canvas = np.full((rows * tile, cols * sheet[0].shape[1], 3), 255, dtype=np.uint8)
            for i, image in enumerate(sheet):
                r, c = divmod(i, cols)
                canvas[r * tile : (r + 1) * tile, c * image.shape[1] : (c + 1) * image.shape[1]] = image
            contact = out / "contact_sheet.png"
            imageio.imwrite(contact, canvas)
            print(f"wrote {contact}")
    finally:
        if close is not None:
            close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--views", type=int, default=24)
    parser.add_argument("--motion-stride", type=int, default=4)
    parser.add_argument(
        "--export-npz",
        type=Path,
        default=None,
        help="Write a NumPy bundle and exit. Use this from the collection env, then render where pytorch3d is installed.",
    )
    args = parser.parse_args()
    if args.export_npz is not None:
        path = export_npz(args.dataset, args.export_npz)
        print(f"wrote {path}")
        return
    out = args.out or args.dataset.parent / "previews"
    render_dataset(args.dataset, out, args.image_size, args.views, args.motion_stride)


if __name__ == "__main__":
    main()
