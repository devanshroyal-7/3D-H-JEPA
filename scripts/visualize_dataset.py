"""Static visualizations of a sharded 6D point-cloud dataset (``outputs/datasets/v1`` layout).

Writes PNG/GIF figures to ``--out``:

- ``pointcloud_gallery_<task>.png``: RGB point clouds at 5 timesteps of one expert and one branched episode.
- ``episode_<task>_<kind>.gif``: point-cloud playback of one episode.
- ``ee_trajectories.png``: top-down end-effector paths, expert vs branched, per task.
- ``workspace_coverage.png``: top-down end-effector visitation density, expert vs branched, per task.
- ``action_distributions_<task>.png``: per-dimension action histograms, expert vs branched.
- ``object_displacement.png``: how far the target object moves per episode, expert vs branched.
- ``dataset_composition.png``: steps per task and store across all shards.

    python scripts/visualize_dataset.py --root outputs/datasets/v1 --out outputs/viz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

TASKS = ("StackCube-v1", "PullCubeTool-v1")
KINDS = ("expert_srmp", "branched_exploratory")
KIND_LABELS = {"expert_srmp": "Expert (SRMP)", "branched_exploratory": "Branched exploratory"}

# Reference light palette: categorical slots 1-2, chrome and ink tokens, blue sequential ramp.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
KIND_COLORS = {"expert_srmp": "#2a78d6", "branched_exploratory": "#eb6834"}
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

WORKSPACE = ((-0.15, 1.05), (-0.55, 0.55), (-0.05, 0.65))
ACTION_NAMES = [f"joint {i + 1} Δ" for i in range(7)] + ["gripper"]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "text.color": TEXT_PRIMARY,
            "font.size": 9,
        }
    )


def _store(root: Path, task: str, kind: str, shard: int):
    return zarr.open_group(str(root / task / f"shard_{shard}" / f"{task}_{kind}.zarr"), mode="r")


def _episode_bounds(g) -> np.ndarray:
    ends = np.asarray(g["episode_ends"], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    return np.stack([starts, ends], axis=1)


def _frame(g, t: int) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(g["xyz"][t], dtype=np.float32), np.asarray(g["rgb"][t], dtype=np.float32) / 255.0


def _scatter_cloud(ax, xyz: np.ndarray, rgb: np.ndarray, title: str) -> None:
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=1.2, linewidths=0, depthshade=False)
    ax.set_xlim(*WORKSPACE[0])
    ax.set_ylim(*WORKSPACE[1])
    ax.set_zlim(*WORKSPACE[2])
    ax.set_box_aspect([r[1] - r[0] for r in WORKSPACE], zoom=1.45)
    ax.view_init(elev=28, azim=-150)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, color=TEXT_SECONDARY, pad=-4)


def pointcloud_gallery(root: Path, out: Path, task: str, episode: int) -> Path:
    fig = plt.figure(figsize=(15, 6.2))
    for row, kind in enumerate(KINDS):
        g = _store(root, task, kind, 0)
        s, e = _episode_bounds(g)[episode]
        ts = np.linspace(s, e - 1, 5).astype(int)
        for col, t in enumerate(ts):
            ax = fig.add_subplot(2, 5, row * 5 + col + 1, projection="3d")
            ax.set_facecolor(SURFACE)
            xyz, rgb = _frame(g, int(t))
            _scatter_cloud(ax, xyz, rgb, f"t = {t - s}")
        fig.text(0.01, 0.75 - row * 0.47, KIND_LABELS[kind], rotation=90, va="center", fontsize=11, color=TEXT_PRIMARY)
    fig.suptitle(f"{task}: 4096-point RGB clouds over one episode (robot-base frame)", x=0.02, ha="left", fontsize=13)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.9, bottom=0.01, wspace=0.0, hspace=0.05)
    path = out / f"pointcloud_gallery_{task}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def episode_gif(root: Path, out: Path, task: str, kind: str, episode: int, stride: int = 2) -> Path:
    g = _store(root, task, kind, 0)
    s, e = _episode_bounds(g)[episode]
    xyz_all = np.asarray(g["xyz"][s:e], dtype=np.float32)
    rgb_all = np.asarray(g["rgb"][s:e], dtype=np.float32) / 255.0
    frames = []
    fig = plt.figure(figsize=(5.5, 4.4))
    for t in range(0, e - s, stride):
        fig.clf()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(SURFACE)
        _scatter_cloud(ax, xyz_all[t], rgb_all[t], "")
        fig.suptitle(f"{task} · {KIND_LABELS[kind]} · t = {t}", fontsize=10, color=TEXT_PRIMARY)
        fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    path = out / f"episode_{task}_{kind}.gif"
    imageio.mimsave(path, frames, duration=0.1, loop=0)
    return path


def _per_episode(g, key: str, max_eps: int) -> list[np.ndarray]:
    arr = np.asarray(g[key])
    return [arr[s:e] for s, e in _episode_bounds(g)[:max_eps]]


def ee_trajectories(root: Path, out: Path, max_eps: int) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True)
    for ax, task in zip(axes, TASKS):
        for kind in reversed(KINDS):  # expert drawn on top
            eps = _per_episode(_store(root, task, kind, 0), "ee_point", max_eps)
            alpha = 0.35 if kind == "expert_srmp" else 0.12
            for i, ep in enumerate(eps):
                ax.plot(ep[:, 0], ep[:, 1], color=KIND_COLORS[kind], lw=0.8, alpha=alpha,
                        label=KIND_LABELS[kind] if i == 0 else None)
        ach = np.concatenate([ep[:1] for ep in _per_episode(_store(root, task, "expert_srmp", 0), "achieved_point", max_eps)])
        ax.scatter(ach[:, 0], ach[:, 1], s=14, facecolors="none", edgecolors=TEXT_PRIMARY, linewidths=0.8,
                   label="Target object start", zorder=5)
        ax.scatter([0], [0], marker="s", s=60, color=TEXT_SECONDARY, label="Robot base", zorder=5)
        ax.set_title(task, loc="left")
        ax.set_xlabel("x (m, robot-base frame)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y (m)")
    fig.suptitle(f"End-effector paths, top-down ({max_eps} episodes per store, shard 0)", x=0.02, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.02, 0.94), ncol=4)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    path = out / "ee_trajectories.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def workspace_coverage(root: Path, out: Path) -> Path:
    bins = [np.linspace(*WORKSPACE[0], 97), np.linspace(*WORKSPACE[1], 89)]
    hists = {}
    for task in TASKS:
        for kind in KINDS:
            ee = np.asarray(_store(root, task, kind, 0)["ee_point"])
            h, _, _ = np.histogram2d(ee[:, 0], ee[:, 1], bins=bins)
            hists[(task, kind)] = h / h.sum()
    vmax = max(h.max() for h in hists.values())
    vmin = min(h[h > 0].min() for h in hists.values())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.6), sharex=True, sharey=True)
    for r, task in enumerate(TASKS):
        for c, kind in enumerate(KINDS):
            ax = axes[r, c]
            h = np.ma.masked_equal(hists[(task, kind)].T, 0)
            im = ax.pcolormesh(bins[0], bins[1], h, cmap=BLUE_RAMP, norm=LogNorm(vmin=vmin, vmax=vmax))
            ax.set_aspect("equal")
            ax.grid(False)
            occupied = 100.0 * (hists[(task, kind)] > 0).mean()
            ax.set_title(f"{task} · {KIND_LABELS[kind]}", loc="left")
            ax.text(0.02, 0.03, f"{occupied:.0f}% of 1.25 cm cells in the crop visited", transform=ax.transAxes,
                    fontsize=8, color=TEXT_SECONDARY)
            if r == 1:
                ax.set_xlabel("x (m)")
            if c == 0:
                ax.set_ylabel("y (m)")
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Fraction of steps (log scale)", color=TEXT_SECONDARY)
    cbar.outline.set_visible(False)
    fig.suptitle("End-effector workspace coverage within the point-cloud crop, top-down (shard 0, all steps)",
                 x=0.02, y=0.93, ha="left", fontsize=13)
    path = out / "workspace_coverage.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def action_distributions(root: Path, out: Path, task: str) -> Path:
    acts = {kind: np.asarray(_store(root, task, kind, 0)["action"]) for kind in KINDS}
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    edges = np.linspace(-1.05, 1.05, 43)
    for d, ax in enumerate(axes.flat):
        for kind in KINDS:
            h, _ = np.histogram(acts[kind][:, d], bins=edges)
            ax.stairs(h / h.sum(), edges, color=KIND_COLORS[kind], lw=2, label=KIND_LABELS[kind])
        ax.set_title(ACTION_NAMES[d], loc="left", fontsize=10)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("normalized action")
    fig.suptitle(f"{task}: pd_joint_delta_pos action distributions (shard 0, share of steps)", x=0.02, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.995), ncol=2)
    path = out / f"action_distributions_{task}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def object_displacement(root: Path, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    edges = np.linspace(0, 0.6, 61)
    for ax, task in zip(axes, TASKS):
        for kind in KINDS:
            eps = _per_episode(_store(root, task, kind, 0), "achieved_point", 10**9)
            disp = np.array([np.linalg.norm(ep[-1] - ep[0]) for ep in eps])
            h, _ = np.histogram(np.clip(disp, 0, edges[-1]), bins=edges)
            moved = 100.0 * (disp > 0.02).mean()
            ax.stairs(h / h.sum(), edges, color=KIND_COLORS[kind], lw=2,
                      label=f"{KIND_LABELS[kind]}: {moved:.0f}% moved > 2 cm")
        ax.set_title(task, loc="left")
        ax.set_xlabel("target object displacement, start → end of episode (m)")
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.legend(loc="upper right")
    fig.suptitle("How much each episode moves the target object (shard 0; last bin includes > 0.6 m)",
                 x=0.02, ha="left", fontsize=13)
    fig.tight_layout()
    path = out / "object_displacement.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def dataset_composition(root: Path, out: Path) -> Path:
    rows = []
    for task in TASKS:
        for kind in KINDS:
            steps = eps = 0
            for store in sorted((root / task).glob(f"shard_*/{task}_{kind}.zarr")):
                ends = np.asarray(zarr.open_group(str(store), mode="r")["episode_ends"])
                eps += len(ends)
                steps += int(ends[-1]) if len(ends) else 0
            rows.append((task, kind, eps, steps))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    labels = [f"{t}\n{KIND_LABELS[k]}" for t, k, _, _ in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[3] / 1e6 for r in rows], color=[KIND_COLORS[r[1]] for r in rows], height=0.62)
    for yi, (_, _, eps, steps) in zip(y, rows):
        ax.text(steps / 1e6 + 0.02, yi, f"{steps / 1e6:.2f}M steps · {eps:,} episodes", va="center",
                fontsize=9, color=TEXT_PRIMARY)
    ax.set_yticks(y, labels)
    ax.set_xlabel("steps (millions)")
    ax.set_xlim(0, max(r[3] for r in rows) / 1e6 * 1.45)
    ax.grid(axis="y", visible=False)
    total = sum(r[3] for r in rows)
    ax.set_title(f"Dataset composition: {total / 1e6:.2f}M steps total, 4 shards per task", loc="left")
    fig.tight_layout()
    path = out / "dataset_composition.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("outputs/datasets/v1"))
    parser.add_argument("--out", type=Path, default=Path("outputs/viz"))
    parser.add_argument("--episode", type=int, default=0, help="Episode index (shard 0) for galleries and GIFs.")
    parser.add_argument("--traj-episodes", type=int, default=150)
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    written = [dataset_composition(args.root, args.out)]
    for task in TASKS:
        written.append(pointcloud_gallery(args.root, args.out, task, args.episode))
        written.append(action_distributions(args.root, args.out, task))
        if not args.no_gif:
            for kind in KINDS:
                written.append(episode_gif(args.root, args.out, task, kind, args.episode))
    written.append(ee_trajectories(args.root, args.out, args.traj_episodes))
    written.append(workspace_coverage(args.root, args.out))
    written.append(object_displacement(args.root, args.out))
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
