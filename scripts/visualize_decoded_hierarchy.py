"""See what every hierarchy level knows and imagines, as point clouds decoded by the per-level PAGE-4D decoders.

For each selected episode, all Level-1 frames are encoded by the frozen world model, every level is built, and each
level's decoder turns three kinds of latents back into clouds:

- ``observed``: the level's own state ``z^(l)_k`` (what survives abstraction),
- ``one-step``: the level predictor's teacher-forced prediction of ``z^(l)_k`` from observed predecessors,
- ``imagined``: an open-loop rollout of the level predictor from its first ``context_len`` states, driven by the
  recorded actions (macro-actions for upper levels).

Interactive (viser): GT | L1 | L2 | ... side by side, with a time slider over Level-1 frames. An upper-level state is
held over the frames until its next state.

    python scripts/visualize_decoded_hierarchy.py \
        --decoder-ckpt outputs/runs/pc_decoder/stageB/decoder_latest.pt \
        --zarr outputs/val_stackcube/StackCube-v1_expert_srmp.zarr --episodes 0 1 --port 8080
    # offline: --no-server --export-gif outputs/viz/decoded.gif --export-ply outputs/viz/decoded_ply
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import zarr

from point_hjepa.data.clip_dataset import _split_episodes
from point_hjepa.diagnostics.decoder_eval import (
    cloud_errors_cm,
    decode_latents,
    encode_episode,
    episode_spans,
    imagine_hierarchy,
    load_episode_l1,
)
from point_hjepa.models.pc_decoder.build import load_decoder_checkpoint
from scripts.train_pc_decoder import load_frozen_world_model

MODES = ("observed", "one-step", "imagined")


def decode_episode(wm, decoder, ep: dict, hier: dict, imag: dict, num_points: int, device) -> dict[str, Any]:
    """Decode every level's observed / one-step / imagined latents of one episode (world frame, numpy)."""
    T1 = ep["xyz"].shape[0]
    gt_xyz = torch.from_numpy(ep["xyz"]).to(device)
    levels = []
    for lvl in range(wm.num_levels):
        idx = hier["l1_indices"][lvl].cpu().numpy()
        T_l = len(idx)
        entry: dict[str, Any] = {"l1_idx": idx, "context": imag["context"][lvl]}
        sources = {
            "observed": (hier["states"][lvl][0], 0),
            "one-step": (None if imag["one_step"][lvl] is None else imag["one_step"][lvl][0], 1),
            "imagined": (None if imag["rollout"][lvl] is None else imag["rollout"][lvl][0], imag["context"][lvl]),
        }
        for mode, (z, first) in sources.items():
            clouds = np.full((T_l, num_points, 6), np.nan, dtype=np.float32)
            err = np.full((T_l,), np.nan, dtype=np.float32)
            valid = np.zeros((T_l,), dtype=bool)
            if z is not None and z.shape[0] > 0:
                pts = decode_latents(decoder, z, lvl, num_points)
                ks = np.arange(first, first + z.shape[0])
                stats = cloud_errors_cm(pts[..., :3], gt_xyz[idx[ks]])
                clouds[ks] = pts.cpu().numpy()
                err[ks] = stats["nn_cm"].cpu().numpy()
                valid[ks] = True
            entry[mode] = {"clouds": clouds, "nn_cm": err, "valid": valid}
        levels.append(entry)
    return {"T1": T1, "gt_xyz": ep["xyz"], "gt_rgb": ep["rgb"], "levels": levels}


def state_at(entry: dict, t: int) -> int:
    """Index of the level state that summarises Level-1 frame ``t`` (the last state at or before ``t``)."""
    return int(np.searchsorted(entry["l1_idx"], t, side="right") - 1)


def summarize(name: str, dec: dict) -> None:
    print(f"\n== {name}: {dec['T1']} Level-1 frames ==")
    for lvl, e in enumerate(dec["levels"]):
        parts = []
        for mode in MODES:
            v = e[mode]["nn_cm"][e[mode]["valid"]]
            parts.append(f"{mode} {v.mean():.2f} cm (n={v.size})" if v.size else f"{mode} -")
        print(f"  L{lvl + 1} ({len(e['l1_idx'])} states, ctx {e['context']}): " + " | ".join(parts))
        im = e["imagined"]
        if im["valid"].any():
            ks = np.flatnonzero(im["valid"])
            h = ks - ks[0] + 1
            marks = sorted({1, max(1, len(h) // 4), max(1, len(h) // 2), len(h)})
            print("      imagined error vs horizon: " + ", ".join(f"h={m}: {im['nn_cm'][ks[m - 1]]:.2f}" for m in marks))


def export_ply(out_dir: Path, name: str, dec: dict) -> None:
    import trimesh

    out_dir.mkdir(parents=True, exist_ok=True)
    for lvl, e in enumerate(dec["levels"]):
        for mode in MODES:
            for k in np.flatnonzero(e[mode]["valid"]):
                c = e[mode]["clouds"][k]
                rgba = np.concatenate([c[:, 3:].clip(0, 255).astype(np.uint8), np.full((len(c), 1), 255, np.uint8)], 1)
                t = int(e["l1_idx"][k])
                trimesh.PointCloud(vertices=c[:, :3], colors=rgba).export(
                    str(out_dir / f"{name}_L{lvl + 1}_{mode}_t{t:04d}.ply")
                )


def export_gif(path: Path, name: str, dec: dict, mode: str, fps: int) -> None:
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    from scripts.visualize_dataset import _scatter_cloud, _style

    _style()
    L = len(dec["levels"])
    frames = []
    for t in range(dec["T1"]):
        fig = plt.figure(figsize=(3.2 * (L + 1), 3.4))
        ax = fig.add_subplot(1, L + 1, 1, projection="3d")
        _scatter_cloud(ax, dec["gt_xyz"][t], dec["gt_rgb"][t] / 255.0, f"GT  t={t}")
        for lvl, e in enumerate(dec["levels"]):
            ax = fig.add_subplot(1, L + 1, lvl + 2, projection="3d")
            k = state_at(e, t)
            m = e[mode]
            if m["valid"][k]:
                c = m["clouds"][k]
                _scatter_cloud(ax, c[:, :3], c[:, 3:].clip(0, 255) / 255.0, f"L{lvl + 1} {mode}  {m['nn_cm'][k]:.2f} cm")
            else:
                c = e["observed"]["clouds"][k]
                _scatter_cloud(ax, c[:, :3], c[:, 3:].clip(0, 255) / 255.0, f"L{lvl + 1} context (observed)")
        fig.suptitle(name, fontsize=10)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, duration=1.0 / max(1, fps), loop=0)
    print(f"wrote {path}")


def serve(decoded: dict[str, dict], host: str, port: int) -> None:
    import viser

    names = list(decoded)
    first = decoded[names[0]]
    L = len(first["levels"])
    gap = 1.3  # metres between panels along -y
    server = viser.ViserServer(host=host, port=port)
    server.scene.set_up_direction("+z")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        mid = -gap * L / 2
        client.camera.position = (2.6, mid - 1.6, 1.8)
        client.camera.look_at = (0.45, mid, 0.15)

    panels = ["GT"] + [f"L{l + 1}" for l in range(L)]
    clouds, labels = [], []
    for i, p in enumerate(panels):
        off = np.array([0.0, -gap * i, 0.0], dtype=np.float32)
        server.scene.add_grid(f"/{p}/grid", width=1.2, height=1.2, plane="xy", cell_size=0.1, position=(0.45, off[1], 0.0))
        clouds.append(
            server.scene.add_point_cloud(
                f"/{p}/cloud", points=first["gt_xyz"][0] + off, colors=first["gt_rgb"][0].astype(np.uint8),
                point_size=0.008, point_shape="circle", precision="float32",
            )
        )
        labels.append(server.scene.add_label(f"/{p}/label", text=p, position=(0.45, off[1], 0.62)))

    ep_dd = server.gui.add_dropdown("Episode", options=names, initial_value=names[0])
    mode_dd = server.gui.add_dropdown("Latent", options=list(MODES), initial_value="observed")
    t_slider = server.gui.add_slider("L1 frame t", min=0, max=max(1, first["T1"] - 1), step=1, initial_value=0)
    play_cb = server.gui.add_checkbox("Play", initial_value=False)
    fps_slider = server.gui.add_slider("FPS", min=1, max=30, step=1, initial_value=6)
    size_slider = server.gui.add_slider("Point size (m)", min=0.003, max=0.025, step=0.001, initial_value=0.008)
    info_md = server.gui.add_markdown("")
    lock = threading.RLock()

    def render() -> None:
        with lock:
            dec = decoded[ep_dd.value]
            t = int(np.clip(t_slider.value, 0, dec["T1"] - 1))
            mode = mode_dd.value
            clouds[0].points = dec["gt_xyz"][t]
            clouds[0].colors = dec["gt_rgb"][t].astype(np.uint8)
            rows = [f"**{ep_dd.value}**, t = {t} / {dec['T1'] - 1}, latent: **{mode}**", ""]
            for lvl, e in enumerate(dec["levels"]):
                k = state_at(e, t)
                m = e[mode]
                src = m if m["valid"][k] else e["observed"]
                c = src["clouds"][k]
                off = np.array([0.0, -gap * (lvl + 1), 0.0], dtype=np.float32)
                clouds[lvl + 1].points = c[:, :3] + off
                clouds[lvl + 1].colors = c[:, 3:].clip(0, 255).astype(np.uint8)
                tag = f"{mode} {m['nn_cm'][k]:.2f} cm" if m["valid"][k] else "context (observed)"
                labels[lvl + 1].text = f"L{lvl + 1} state {k} (t={int(e['l1_idx'][k])}): {tag}"
                rows.append(f"- L{lvl + 1}: state {k}, {tag}")
            for cl in clouds:
                cl.point_size = float(size_slider.value)
            info_md.content = "\n".join(rows)

    @ep_dd.on_update
    def _(_: Any) -> None:
        t_slider.max = max(1, decoded[ep_dd.value]["T1"] - 1)
        t_slider.value = 0
        render()

    for h in (mode_dd, t_slider, size_slider):
        h.on_update(lambda _: render())
    render()
    print(f"viser at http://{host}:{port}  (Ctrl+C to stop)")
    while True:
        if play_cb.value:
            T1 = decoded[ep_dd.value]["T1"]
            t_slider.value = (int(t_slider.value) + 1) % T1
            time.sleep(1.0 / float(fps_slider.value))
        else:
            time.sleep(0.05)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decoder-ckpt", type=Path, required=True)
    ap.add_argument("--wm-ckpt", type=Path, default=None, help="default: the WM the decoder was trained on")
    ap.add_argument("--zarr", type=Path, nargs="+", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="episode ids per store (default: val split)")
    ap.add_argument("--max-episodes", type=int, default=2, help="episodes per store when --episodes is not given")
    ap.add_argument("--max-l1-frames", type=int, default=None, help="truncate long episodes (Level-1 frames)")
    ap.add_argument("--context", type=int, nargs="*", default=None, help="rollout context per level (default ctx len)")
    ap.add_argument("--num-points", type=int, default=4096)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--export-gif", type=Path, default=None, help="GIF per episode (suffix _<name>_<mode>.gif)")
    ap.add_argument("--gif-mode", choices=MODES, default="imagined")
    ap.add_argument("--gif-fps", type=int, default=6)
    ap.add_argument("--export-ply", type=Path, default=None)
    ap.add_argument("--no-server", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder, dckpt = load_decoder_checkpoint(args.decoder_ckpt, device=device)
    decoder.eval()
    wm, wm_cfg = load_frozen_world_model(Path(args.wm_ckpt or dckpt["wm_checkpoint"]), device)
    if decoder.num_levels != wm.num_levels:
        raise ValueError(f"decoder has {decoder.num_levels} levels, world model {wm.num_levels}")

    decoded: dict[str, dict] = {}
    for ds_idx, path in enumerate(args.zarr):
        root = zarr.open_group(str(path), mode="r")
        spans = episode_spans(root)
        if args.episodes:
            ep_ids = list(args.episodes)
        else:
            ends = np.asarray(root["episode_ends"][:], dtype=np.int64)
            ep_ids = sorted(_split_episodes(ends, args.val_ratio, "val", args.seed + ds_idx))[: args.max_episodes]
        for ep_id in ep_ids:
            s, e = spans[ep_id]
            if args.max_l1_frames is not None:
                e = min(e, s + (args.max_l1_frames - 1) * int(wm_cfg.frame_skip) + 1)
            ep = load_episode_l1(root, s, e, int(wm_cfg.frame_skip), int(wm_cfg.action_dim))
            hier = encode_episode(
                wm, ep, device, preset=str(wm_cfg.backbone.preset), scale=float(wm_cfg.backbone.scale),
                center=bool(wm_cfg.backbone.center),
            )
            imag = imagine_hierarchy(wm, hier, args.context)
            name = f"{path.stem}/ep{ep_id}"
            decoded[name] = decode_episode(wm, decoder, ep, hier, imag, args.num_points, device)
            summarize(name, decoded[name])
            safe = name.replace("/", "_")
            if args.export_ply is not None:
                export_ply(args.export_ply, safe, decoded[name])
            if args.export_gif is not None:
                gif = args.export_gif.with_name(f"{args.export_gif.stem}_{safe}_{args.gif_mode}.gif")
                export_gif(gif, name, decoded[name], args.gif_mode, args.gif_fps)

    if not args.no_server:
        serve(decoded, args.host, args.port)


if __name__ == "__main__":
    main()
