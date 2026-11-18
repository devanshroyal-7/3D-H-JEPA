"""Point-cloud preprocessing for 3D Point H-JEPA.

Pipeline, applied per frame (on GPU when CUDA is available):

1. Drop invalid points (`xyzw[..., 3] == 0`, the ManiSkill homogeneous flag
   for rays that miss before the camera far plane).
2. Transform the remaining points from the world frame into the robot base frame.
3. Crop to a workspace axis-aligned box.
4. Use the segmentation ids to keep every robot and task-object point, plus a
   sparse random slice of the table. Ground and background are dropped.
5. Batched farthest-point sampling, or voxel pooling, down to a fixed `(N, 3)`.
   An optional RGB channel is gathered with the same indices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from point_hjepa.data.geometry import quat_to_mat


@dataclass
class PreprocessConfig:
    num_points: int = 2048
    workspace_min: tuple[float, float, float] = (-0.2, -0.7, -0.05)
    workspace_max: tuple[float, float, float] = (1.0, 0.7, 0.8)
    table_keep: int = 256
    downsample: str = "fps"  # fps | voxel
    voxel_size: float = 0.01
    store_rgb: bool = False
    fps_pre_cap: int = 4096


def _as_batched_torch(value, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def farthest_point_sample(points: torch.Tensor, npoint: int, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Batched farthest-point sampling.

    `points` is `(B, M, 3)`. `valid` is an optional `(B, M)` boolean mask.
    Returns long indices `(B, npoint)`. Rows with fewer valid points repeat
    their first valid index so the output length stays fixed.
    """
    if points.ndim != 3:
        raise ValueError(f"points must be (B, M, 3), got {tuple(points.shape)}")
    batch, count, _ = points.shape
    npoint = int(npoint)
    device = points.device
    if valid is None:
        valid = torch.ones(batch, count, dtype=torch.bool, device=device)
    if count == 0 or npoint <= 0:
        return torch.zeros(batch, max(npoint, 0), dtype=torch.long, device=device)

    # Work in numpy on CPU: the python loop is cheaper than a launch per iteration
    # for the cloud sizes left after cropping. CUDA keeps the torch loop.
    if device.type == "cpu":
        return _fps_numpy(points.detach().cpu().numpy(), npoint, valid.detach().cpu().numpy())

    centroids = torch.zeros(batch, npoint, dtype=torch.long, device=device)
    distance = torch.full((batch, count), 1e10, device=device)
    distance = torch.where(valid, distance, torch.full_like(distance, -1.0))
    first = valid.to(torch.int64).argmax(dim=1)
    none_valid = ~valid.any(dim=1)
    first = torch.where(none_valid, torch.zeros_like(first), first)
    batch_idx = torch.arange(batch, device=device)
    farthest = first
    for i in range(npoint):
        centroids[:, i] = farthest
        chosen = points[batch_idx, farthest].unsqueeze(1)
        dist = ((points - chosen) ** 2).sum(-1)
        dist = torch.where(valid, dist, torch.full_like(dist, -1.0))
        closer = (dist < distance) & valid & (dist >= 0)
        distance = torch.where(closer, dist, distance)
        farthest = distance.argmax(dim=1)
        farthest = torch.where(none_valid, torch.zeros_like(farthest), farthest)
    return centroids


def _fps_numpy(points: np.ndarray, npoint: int, valid: np.ndarray) -> torch.Tensor:
    batch, count, _ = points.shape
    centroids = np.zeros((batch, npoint), dtype=np.int64)
    for b in range(batch):
        ok = np.flatnonzero(valid[b])
        if ok.size == 0:
            continue
        if ok.size <= npoint:
            take = np.resize(ok, npoint)
            centroids[b] = take
            continue
        chosen = np.empty(npoint, dtype=np.int64)
        subset = points[b, ok]
        dist = np.full(ok.size, 1e10, dtype=np.float64)
        farthest = 0
        for i in range(npoint):
            chosen[i] = ok[farthest]
            delta = subset - subset[farthest]
            d = np.einsum("ij,ij->i", delta, delta)
            dist = np.minimum(dist, d)
            farthest = int(np.argmax(dist))
        centroids[b] = chosen
    return torch.from_numpy(centroids)


def voxel_downsample(
    points: np.ndarray, rgb: np.ndarray | None, voxel_size: float
) -> tuple[np.ndarray, np.ndarray | None]:
    if len(points) == 0:
        return points, rgb
    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    index.sort()
    rgb_out = None if rgb is None else rgb[index]
    return points[index], rgb_out


def _pad_or_sample(points: np.ndarray, rgb: np.ndarray | None, npoint: int, rng: np.random.Generator):
    count = len(points)
    if count == 0:
        zeros = np.zeros((npoint, 3), dtype=np.float32)
        rgb_out = None if rgb is None else np.zeros((npoint, 3), dtype=np.uint8)
        return zeros, rgb_out, 0
    if count == npoint:
        return points.astype(np.float32), None if rgb is None else rgb, count
    if count > npoint:
        raise RuntimeError("caller should downsample before padding")
    extra = rng.choice(count, size=npoint - count, replace=True)
    index = np.concatenate([np.arange(count), extra])
    rgb_out = None if rgb is None else rgb[index]
    return points[index].astype(np.float32), rgb_out, count


def preprocess_pointclouds(
    xyzw,
    segmentation,
    *,
    base_p: np.ndarray,
    base_q: np.ndarray,
    robot_ids: set[int] | list[int],
    object_ids: set[int] | list[int],
    table_ids: set[int] | list[int],
    config: PreprocessConfig | None = None,
    rgb=None,
    rng: np.random.Generator | None = None,
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Preprocess a batch of fused point clouds.

    `xyzw` is `(M, 4)` or `(B, M, 4)`. Segmentation is `(M, 1)` or `(B, M, 1)`
    (or the same without the trailing singleton). Returns `points` of shape
    `(B, N, 3)` float32, `point_count` `(B,)` int32, and `rgb` `(B, N, 3)`
    uint8 when requested.
    """
    cfg = config or PreprocessConfig()
    rng = rng or np.random.default_rng(0)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        torch_device = torch.device("cpu")

    xyzw_t = _as_batched_torch(xyzw, torch_device).to(torch.float32)
    seg_t = _as_batched_torch(segmentation, torch_device)
    if seg_t.ndim == 3 and seg_t.shape[-1] == 1:
        seg_t = seg_t[..., 0]
    rgb_t = None
    if cfg.store_rgb:
        if rgb is None:
            raise ValueError("store_rgb is set but no rgb array was passed")
        rgb_t = _as_batched_torch(rgb, torch_device)

    batch = xyzw_t.shape[0]
    rot = torch.as_tensor(quat_to_mat(base_q), dtype=torch.float32, device=torch_device)
    origin = torch.as_tensor(base_p, dtype=torch.float32, device=torch_device)
    wmin = torch.as_tensor(cfg.workspace_min, dtype=torch.float32, device=torch_device)
    wmax = torch.as_tensor(cfg.workspace_max, dtype=torch.float32, device=torch_device)
    robot = {int(i) for i in robot_ids}
    objects = {int(i) for i in object_ids}
    table = {int(i) for i in table_ids}
    keep_ids = robot | objects

    clouds: list[np.ndarray] = []
    colors: list[np.ndarray | None] = []
    for b in range(batch):
        pts = xyzw_t[b]
        valid = torch.isfinite(pts[:, :3]).all(dim=-1) & (pts[:, 3] > 0.5)
        xyz = pts[:, :3][valid]
        seg = seg_t[b][valid].to(torch.int64)
        col = None if rgb_t is None else rgb_t[b][valid]
        if xyz.shape[0] == 0:
            clouds.append(np.zeros((0, 3), dtype=np.float32))
            colors.append(None if col is None else np.zeros((0, 3), dtype=np.uint8))
            continue
        xyz = (xyz - origin) @ rot
        inside = ((xyz >= wmin) & (xyz <= wmax)).all(dim=-1)
        xyz = xyz[inside]
        seg = seg[inside]
        if col is not None:
            col = col[inside]
        if xyz.shape[0] == 0:
            clouds.append(np.zeros((0, 3), dtype=np.float32))
            colors.append(None if col is None else np.zeros((0, 3), dtype=np.uint8))
            continue
        seg_np = seg.detach().cpu().numpy()
        xyz_np = xyz.detach().cpu().numpy().astype(np.float32)
        col_np = None if col is None else col.detach().cpu().numpy()
        keep = np.isin(seg_np, np.fromiter(keep_ids, dtype=np.int64, count=len(keep_ids))) if keep_ids else np.zeros(len(seg_np), dtype=bool)
        on_table = np.isin(seg_np, np.fromiter(table, dtype=np.int64, count=len(table))) if table else np.zeros(len(seg_np), dtype=bool)
        obj_xyz = xyz_np[keep]
        obj_rgb = None if col_np is None else col_np[keep]
        tab_xyz = xyz_np[on_table]
        tab_rgb = None if col_np is None else col_np[on_table]
        if len(tab_xyz) > cfg.table_keep:
            choice = rng.choice(len(tab_xyz), size=cfg.table_keep, replace=False)
            tab_xyz = tab_xyz[choice]
            if tab_rgb is not None:
                tab_rgb = tab_rgb[choice]
        if len(obj_xyz) and len(tab_xyz):
            merged = np.concatenate([obj_xyz, tab_xyz], axis=0)
            merged_rgb = None if obj_rgb is None else np.concatenate([obj_rgb, tab_rgb], axis=0)
        elif len(obj_xyz):
            merged, merged_rgb = obj_xyz, obj_rgb
        else:
            merged, merged_rgb = tab_xyz, tab_rgb
        clouds.append(merged.astype(np.float32))
        colors.append(None if merged_rgb is None else merged_rgb.astype(np.uint8))

    npoint = int(cfg.num_points)
    out_pts = np.zeros((batch, npoint, 3), dtype=np.float32)
    out_rgb = np.zeros((batch, npoint, 3), dtype=np.uint8) if cfg.store_rgb else None
    counts = np.zeros(batch, dtype=np.int32)

    # Voxel-pool first when requested, then FPS anything still above N.
    processed: list[tuple[np.ndarray, np.ndarray | None]] = []
    for xyz, col in zip(clouds, colors):
        if cfg.downsample == "voxel":
            xyz, col = voxel_downsample(xyz, col, cfg.voxel_size)
        if len(xyz) > cfg.fps_pre_cap:
            choice = rng.choice(len(xyz), size=cfg.fps_pre_cap, replace=False)
            xyz = xyz[choice]
            if col is not None:
                col = col[choice]
        processed.append((xyz, col))

    need_fps = [i for i, (xyz, _) in enumerate(processed) if len(xyz) > npoint]
    fps_index: dict[int, np.ndarray] = {}
    if need_fps and cfg.downsample in ("fps", "voxel"):
        width = max(len(processed[i][0]) for i in need_fps)
        buf = np.zeros((len(need_fps), width, 3), dtype=np.float32)
        mask = np.zeros((len(need_fps), width), dtype=bool)
        for row, i in enumerate(need_fps):
            xyz = processed[i][0]
            buf[row, : len(xyz)] = xyz
            mask[row, : len(xyz)] = True
        sampled = (
            farthest_point_sample(
                torch.from_numpy(buf).to(torch_device),
                npoint,
                torch.from_numpy(mask).to(torch_device),
            )
            .detach()
            .cpu()
            .numpy()
        )
        for row, i in enumerate(need_fps):
            fps_index[i] = sampled[row]

    for i, (xyz, col) in enumerate(processed):
        if i in fps_index:
            index = fps_index[i]
            # Indices were computed on the padded buffer; they only point at real points.
            taken = xyz[index]
            taken_rgb = None if col is None else col[index]
            out_pts[i] = taken.astype(np.float32)
            if out_rgb is not None and taken_rgb is not None:
                out_rgb[i] = taken_rgb
            counts[i] = npoint
            continue
        padded, padded_rgb, count = _pad_or_sample(xyz, col, npoint, rng)
        out_pts[i] = padded
        if out_rgb is not None and padded_rgb is not None:
            out_rgb[i] = padded_rgb
        counts[i] = count

    result = {"points": out_pts, "point_count": counts}
    if out_rgb is not None:
        result["rgb"] = out_rgb
    return result
