"""Train per-level JEPA4R point-cloud decoders (PAGE-4D trunk) on a frozen Stage-1 Point-H-JEPA world model.

Every level ``l`` gets its own decoder (FiLM set generator + affine frame + rgb branch, + per-level LoRA in stage B)
on one shared frozen PAGE-4D trunk. Level-``l`` state ``k`` is supervised with the observed cloud of the Level-1
frame it summarises (``PointHJEPA.build_hierarchy`` -> ``l1_indices``).

Stage A: python scripts/train_pc_decoder.py --config-name pc_decoder
Stage B: python scripts/train_pc_decoder.py --config-name pc_decoder decoder.stage=B \
           decoder.init_from=outputs/runs/pc_decoder/stageA/decoder_latest.pt
DDP:     torchrun --nproc_per_node=4 scripts/train_pc_decoder.py --config-name pc_decoder
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import nullcontext
from itertools import cycle
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from point_hjepa.data.clip_dataset import HierarchicalClipDataset, make_ptv3_collate_fn
from point_hjepa.diagnostics.decoder_eval import cloud_errors_cm
from point_hjepa.diagnostics.eval_probes import move_point_to_device
from point_hjepa.models.pc_decoder.build import (
    build_trunk,
    decoder_kwargs,
    estimate_workspace_center,
    load_decoder_checkpoint,
    save_decoder_checkpoint,
)
from point_hjepa.models.pc_decoder.hierarchical_decoder import HierarchicalPointDecoder
from point_hjepa.models.point_hjepa import PointHJEPA, compute_min_l1_frames
from scripts.train_ddp import _cosine_lr_factor, _init_distributed, build_model_from_cfg, init_wandb_run

log = logging.getLogger(__name__)


def load_frozen_world_model(ckpt_path: Path, device: torch.device) -> tuple[PointHJEPA, DictConfig]:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise KeyError(f"{ckpt_path} has no 'config'; cannot rebuild the world model")
    wm_cfg = OmegaConf.create(ckpt["config"])
    wm = build_model_from_cfg(wm_cfg)
    wm.load_state_dict(ckpt["model_state_dict"], strict=True)
    wm.to(device).eval().requires_grad_(False)
    return wm, wm_cfg


def wm_num_l1_frames(wm_cfg: DictConfig) -> int:
    min_l1 = compute_min_l1_frames(list(wm_cfg.strides), list(wm_cfg.context_lengths))
    n = wm_cfg.get("num_l1_frames", None)
    return max(min_l1, int(n)) if n is not None else min_l1


@torch.no_grad()
def gather_level_frames(
    wm: PointHJEPA,
    batch: dict[str, Any],
    device: torch.device,
    frames_per_level: int | None,
    amp_ctx: Any,
    generator: torch.Generator | None = None,
    with_one_step: bool = False,
) -> list[dict[str, torch.Tensor]]:
    """Encode a clip batch with the frozen WM and gather, per level, latents + the GT cloud of their frame.

    Returns one dict per level: ``z [n, D]``, ``xyz [n, N, 3]``, ``rgb [n, N, 3]`` (and, with ``with_one_step``,
    ``z_pred [m, D]`` + ``xyz_pred`` / ``rgb_pred``: teacher-forced one-step predictions of states ``1..``).
    """
    B, T = int(batch["batch_size"]), int(batch["seq_len"])
    point = move_point_to_device(batch["point"], device)
    actions = batch["actions"].to(device, non_blocking=True)
    xyz = batch["xyz"].to(device, non_blocking=True)  # [B, T, N, 3]
    rgb = batch["rgb"].to(device, non_blocking=True)
    with amp_ctx:
        hier = wm.encode_hierarchy(point, actions, batch_size=B, seq_len=T)
        preds = (
            [wm.predictors[l](hier["states"][l][:, :-1], hier["actions"][l]) for l in range(wm.num_levels)]
            if with_one_step
            else None
        )

    out = []
    for lvl in range(wm.num_levels):
        idx = hier["l1_indices"][lvl]
        z = hier["states"][lvl].float().reshape(-1, wm.latent_dim)  # [B * T_l, D]
        gx = xyz[:, idx].reshape(-1, *xyz.shape[2:])
        gr = rgb[:, idx].reshape(-1, *rgb.shape[2:])
        if frames_per_level is not None and z.shape[0] > frames_per_level:
            sel = torch.randperm(z.shape[0], device=device, generator=generator)[:frames_per_level]
            z, gx, gr = z[sel], gx[sel], gr[sel]
        item = {"z": z, "xyz": gx, "rgb": gr}
        if preds is not None:
            item["z_pred"] = preds[lvl].float().reshape(-1, wm.latent_dim)
            item["xyz_pred"] = xyz[:, idx[1:]].reshape(-1, *xyz.shape[2:])
            item["rgb_pred"] = rgb[:, idx[1:]].reshape(-1, *rgb.shape[2:])
        out.append(item)
    return out


def subsample_points(xyz: torch.Tensor, rgb: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A random ``n``-point subset per frame (``[F, N, 3]`` -> ``[F, n, 3]``)."""
    if n >= xyz.shape[1]:
        return xyz, rgb
    idx = torch.rand(xyz.shape[0], xyz.shape[1], device=xyz.device).argsort(dim=1)[:, :n]
    g = idx.unsqueeze(-1).expand(-1, -1, 3)
    return torch.gather(xyz, 1, g), torch.gather(rgb, 1, g)


@torch.no_grad()
def validate(
    wm: PointHJEPA,
    decoder: HierarchicalPointDecoder,
    loader: DataLoader,
    dcfg: DictConfig,
    device: torch.device,
    amp_ctx_fn: Any,
    max_batches: int,
    num_clouds: int = 0,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Per-level NN error (cm) of decoded observed latents and of decoded one-step predictions."""
    decoder.eval()
    n_pts = int(dcfg.num_points_eval)
    acc: dict[str, list[torch.Tensor]] = {}
    clouds: dict[str, Any] = {}
    gen = torch.Generator(device=device).manual_seed(0)
    for b_idx, batch in enumerate(loader):
        if b_idx >= max_batches:
            break
        levels = gather_level_frames(
            wm, batch, device, int(dcfg.frames_per_level), amp_ctx_fn(), generator=gen, with_one_step=True
        )
        for lvl, item in enumerate(levels):
            for kind, zk, xk, rk in (
                ("obs", item["z"], item["xyz"], item["rgb"]),
                ("pred1", item["z_pred"], item["xyz_pred"], item["rgb_pred"]),
            ):
                with amp_ctx_fn():
                    pts = decoder.decode_points(zk, lvl, n_pts).float()
                stats = cloud_errors_cm(pts[..., :3], xk.float(), float(dcfg.outlier_cm))
                for k, v in stats.items():
                    acc.setdefault(f"l{lvl + 1}/{kind}_{k}", []).append(v)
                if b_idx == 0 and kind == "obs" and num_clouds > 0:
                    for i in range(min(num_clouds, pts.shape[0])):
                        clouds[f"l{lvl + 1}/frame{i}_decoded"] = pts[i].cpu().numpy()
                        clouds[f"l{lvl + 1}/frame{i}_gt"] = torch.cat([xk[i].float(), rk[i].float()], -1).cpu().numpy()
    metrics = {f"val/{k}": float(torch.cat(v).mean().item()) for k, v in acc.items()}
    decoder.train()
    return metrics, clouds


@hydra.main(config_path="../configs", config_name="pc_decoder", version_base=None)
def main(cfg: DictConfig) -> None:
    rank, local_rank, world_size, device = _init_distributed()
    is_main = rank == 0
    torch.manual_seed(int(cfg.train.seed) + rank)
    dcfg = cfg.decoder
    stage = str(dcfg.stage).upper()
    if stage not in ("A", "B"):
        raise ValueError(f"decoder.stage must be A or B, got {dcfg.stage!r}")

    out_dir = Path(dcfg.output_dir) / f"stage{stage}"
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info("Decoder config:\n%s", OmegaConf.to_yaml(dcfg))

    # ---- frozen world model (its own config defines the hierarchy and the point preprocessing)
    wm, wm_cfg = load_frozen_world_model(Path(dcfg.wm_checkpoint), device)
    num_l1 = wm_num_l1_frames(wm_cfg)

    # ---- data: the decoder run's datasets (cfg.datasets) at the WM's clip length / frame skip
    sources = [OmegaConf.to_container(d, resolve=True) for d in cfg.datasets]
    ds_kw = dict(
        frame_skip=int(wm_cfg.frame_skip),
        num_l1_frames=num_l1,
        action_dim=int(wm_cfg.action_dim),
        val_ratio=float(cfg.val_split),
        seed=int(cfg.train.seed),
    )
    train_ds = HierarchicalClipDataset(sources, split="train", **ds_kw)
    val_ds = HierarchicalClipDataset(sources, split="val", **ds_kw)
    collate_fn = make_ptv3_collate_fn(
        preset=str(wm_cfg.backbone.preset),
        scale=float(wm_cfg.backbone.scale),
        center=bool(wm_cfg.backbone.center),
        keep_raw=True,
    )
    clips = int(dcfg.clips_per_step)
    train_sampler = train_ds.make_distributed_weighted_sampler(
        num_replicas=world_size, rank=rank, seed=int(cfg.train.seed), drop_last=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=clips,
        sampler=train_sampler,
        num_workers=int(dcfg.num_workers),
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=int(dcfg.num_workers) > 0,
    )
    val_gen = torch.Generator().manual_seed(0)
    val_loader = DataLoader(
        val_ds,
        batch_size=clips,
        sampler=torch.utils.data.RandomSampler(val_ds, generator=val_gen),
        num_workers=int(dcfg.num_workers),
        collate_fn=collate_fn,
        drop_last=False,
    )

    # ---- decoder
    if stage == "B":
        if not dcfg.init_from:
            raise ValueError("Stage B needs decoder.init_from=<stage-A decoder checkpoint>")
        decoder, init_ckpt = load_decoder_checkpoint(dcfg.init_from)
        target_center = [float(c) for c in decoder.target_center.tolist()]
        decoder.enable_lora(int(dcfg.lora_rank))
        if is_main:
            log.info("Stage B: initialised from %s (%d tensors)", dcfg.init_from, len(init_ckpt["decoder_state_dict"]))
    else:
        if str(dcfg.target_center) == "auto":
            target_center = estimate_workspace_center([s["path"] for s in sources], seed=int(cfg.train.seed))
        else:
            target_center = [float(c) for c in dcfg.target_center]
        decoder = HierarchicalPointDecoder(
            build_trunk(dcfg), **decoder_kwargs(dcfg, wm.num_levels, wm.latent_dim, target_center)
        )
    decoder.to(device).train()
    if is_main:
        log.info("target frame: center=%s scale=%.2f", target_center, float(decoder.target_scale))
    # the architecture is the one actually built (stage B: stage A's), so checkpoints always reload
    kwargs = (
        dict(init_ckpt["decoder_kwargs"])
        if stage == "B"
        else decoder_kwargs(dcfg, wm.num_levels, wm.latent_dim, target_center)
    )

    groups = [{"name": "new", "params": decoder.head_parameters(), "lr": float(dcfg.lr)}]
    if stage == "B":
        groups.append({"name": "lora", "params": decoder.lora_parameters(), "lr": float(dcfg.lora_lr)})
    optimizer = torch.optim.AdamW(
        groups, weight_decay=float(dcfg.weight_decay), betas=tuple(float(b) for b in dcfg.betas)
    )
    total_steps = int(dcfg.steps)
    warmup_steps = max(1, int(round(float(dcfg.warmup_ratio) * total_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _cosine_lr_factor(s, total_steps, warmup_steps)
    )

    model: torch.nn.Module = decoder
    if world_size > 1:
        model = DDP(decoder, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=False)

    precision = str(cfg.train.precision).lower()
    use_amp = device.type == "cuda" and precision == "bf16"

    def amp_ctx() -> Any:
        return torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    trainable = sum(p.numel() for g in groups for p in g["params"])
    runtime_meta = {
        "world_size": world_size,
        "stage": stage,
        "num_levels": wm.num_levels,
        "num_l1_frames": num_l1,
        "frames_per_step": int(dcfg.frames_per_level) * wm.num_levels * world_size,
        "trainable_params": trainable,
        "train_clips": len(train_ds),
        "val_clips": len(val_ds),
        "target_center": target_center,
    }
    if is_main:
        log.info("runtime: %s", runtime_meta)
    if "wandb" in cfg and bool(cfg.wandb.enabled):
        cfg.wandb.job_type = f"pc_decoder_stage{stage}"
    wandb_run = init_wandb_run(cfg, is_main=is_main, out_dir=out_dir, runtime_meta=runtime_meta)

    meta = {
        "stage": stage,
        "decoder_kwargs": kwargs,
        "decoder_config": OmegaConf.to_container(dcfg, resolve=True),
        "lora_rank": int(dcfg.lora_rank) if stage == "B" else 0,
        "wm_checkpoint": str(dcfg.wm_checkpoint),
        "wm_config": OmegaConf.to_container(wm_cfg, resolve=True),
    }

    def run_val(step: int) -> None:
        if not is_main:
            return
        metrics, clouds = validate(
            wm, decoder, val_loader, dcfg, device, amp_ctx, int(dcfg.val_batches), int(dcfg.wandb_clouds)
        )
        log.info("step=%d val: %s", step, {k: round(v, 3) for k, v in metrics.items() if "nn_cm" in k})
        if wandb_run is not None:
            import wandb

            payload: dict[str, Any] = dict(metrics)
            for k, arr in clouds.items():
                pts = arr.copy()
                pts[:, 3:] = pts[:, 3:].clip(0, 255)
                payload[f"clouds/{k}"] = wandb.Object3D(pts)
            wandb_run.log(payload, step=step)
        save_decoder_checkpoint(out_dir / "decoder_latest.pt", decoder, {**meta, "global_step": step, "val": metrics})

    exit_code = 0
    n_pts_train = int(dcfg.num_points_train)
    try:
        step, epoch = 0, 0
        t0 = time.perf_counter()
        while step < total_steps:
            epoch += 1
            train_sampler.set_epoch(epoch)
            for batch in train_loader:
                if step >= total_steps:
                    break
                levels = gather_level_frames(wm, batch, device, int(dcfg.frames_per_level), amp_ctx())
                z_all = torch.cat([it["z"] for it in levels])
                lvl_all = torch.cat(
                    [torch.full((it["z"].shape[0],), l, dtype=torch.long, device=device) for l, it in enumerate(levels)]
                )
                targets = []
                for it in levels:
                    gx, gr = subsample_points(it["xyz"].float(), it["rgb"].float(), n_pts_train)
                    targets.append(decoder.to_target(gx, gr))
                target_all = torch.cat(targets)

                optimizer.zero_grad(set_to_none=True)
                with amp_ctx():
                    dense = model(z_all, lvl_all)
                loss = dense.new_zeros(())
                logs: dict[str, torch.Tensor] = {}
                for l in range(wm.num_levels):
                    m = lvl_all == l
                    loss_l, terms = decoder.loss(dense[m], target_all[m], n_pts_train)
                    loss = loss + loss_l / wm.num_levels
                    logs[f"l{l + 1}/loss"] = loss_l.detach()
                    for k, v in terms.items():
                        logs[f"l{l + 1}/{k}"] = v
                logs["loss"] = loss.detach()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for g in groups for p in g["params"]], float(dcfg.grad_clip)
                )
                optimizer.step()
                scheduler.step()
                step += 1

                if step == 1 or step % int(dcfg.log_every) == 0:
                    keys = sorted(logs)
                    vals = torch.stack([logs[k].float() for k in keys])
                    if world_size > 1:
                        dist.all_reduce(vals, op=dist.ReduceOp.SUM)
                        vals = vals / world_size
                    if is_main:
                        dt = (time.perf_counter() - t0) / (1 if step == 1 else int(dcfg.log_every))
                        t0 = time.perf_counter()
                        d = {f"train/{k}": float(v) for k, v in zip(keys, vals.tolist())}
                        d["train/grad_norm"] = float(grad_norm)
                        d["train/step_time_s"] = dt
                        for g in optimizer.param_groups:
                            d[f"train/lr_{g['name']}"] = float(g["lr"])
                        log.info("stage=%s step=%d loss=%.4f (%.2fs/step)", stage, step, d["train/loss"], dt)
                        if wandb_run is not None:
                            wandb_run.log(d, step=step)

                if step % int(dcfg.val_every) == 0 and step < total_steps:
                    run_val(step)
                    if world_size > 1:
                        dist.barrier()
        run_val(step)
        if world_size > 1:
            dist.barrier()
    except BaseException:
        exit_code = 1
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish(exit_code=exit_code)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
