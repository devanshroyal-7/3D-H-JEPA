"""Stage-1 DDP training driver for Point-H-JEPA (4x H100 or single-GPU fallback)."""

from __future__ import annotations

import logging
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from point_hjepa.data.clip_dataset import HierarchicalClipDataset, make_ptv3_collate_fn
from point_hjepa.diagnostics.eval_probes import move_point_to_device, run_epoch_diagnostics
from point_hjepa.models.point_hjepa import PointHJEPA, compute_min_l1_frames

log = logging.getLogger(__name__)


def _init_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, local_rank, world_size, device


def build_model_from_cfg(cfg: DictConfig) -> PointHJEPA:
    pred_max_seq_len = cfg.predictor.get("max_seq_len", None)
    if pred_max_seq_len is None and cfg.get("num_l1_frames", None) is not None:
        min_l1_frames = compute_min_l1_frames(list(cfg.strides), list(cfg.context_lengths))
        pred_max_seq_len = max(64, min_l1_frames, int(cfg.num_l1_frames))
    return PointHJEPA(
        num_levels=int(cfg.num_levels),
        strides=list(cfg.strides),
        context_lengths=list(cfg.context_lengths),
        action_dim=int(cfg.action_dim),
        frame_skip=int(cfg.frame_skip),
        latent_dim=int(cfg.latent_dim),
        action_l1_dim=int(cfg.action_l1_dim),
        macro_action_dim=int(cfg.macro_action_dim),
        action_pooler_type=str(cfg.action_pooler_type),
        backbone_preset=str(cfg.backbone.preset),
        backbone_pretrained=bool(cfg.backbone.pretrained),
        serialization_depth=int(cfg.backbone.serialization_depth),
        freeze_stages=int(cfg.backbone.freeze_stages),
        fourier_bands=int(cfg.backbone.fourier_bands),
        utonia_scale=float(cfg.backbone.scale),
        pooler_queries=int(cfg.pooler.num_queries),
        pooler_heads=int(cfg.pooler.num_heads),
        pooler_layers=int(cfg.pooler.num_layers),
        pooler_mlp_ratio=float(cfg.pooler.mlp_ratio),
        predictor_depth=int(cfg.predictor.depth),
        predictor_heads=int(cfg.predictor.num_heads),
        predictor_mlp_ratio=float(cfg.predictor.mlp_ratio),
        predictor_max_seq_len=None if pred_max_seq_len is None else int(pred_max_seq_len),
        lambda_sigreg=list(cfg.loss.lambda_sigreg),
        gamma_idm=list(cfg.loss.gamma_idm),
        sigreg_num_slices=int(cfg.loss.sigreg_num_slices),
        sigreg_t_min=float(cfg.loss.sigreg_t_min),
        sigreg_t_max=float(cfg.loss.sigreg_t_max),
        sigreg_num_knots=int(cfg.loss.sigreg_num_knots),
    )


def _cosine_lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def init_wandb_run(
    cfg: DictConfig,
    *,
    is_main: bool,
    out_dir: Path,
    runtime_meta: dict[str, Any],
) -> Any:
    """Initialize a W&B run on rank 0 with resolved config and runtime metadata."""
    if not is_main or not bool(cfg.wandb.enabled):
        return None
    import wandb

    raw_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb_cfg: dict[str, Any] = dict(raw_cfg) if isinstance(raw_cfg, dict) else {"cfg": raw_cfg}
    wandb_cfg["runtime"] = dict(runtime_meta)

    entity = cfg.wandb.get("entity", None)
    name = cfg.wandb.get("name", None)
    group = cfg.wandb.get("group", None)
    raw_tags = cfg.wandb.get("tags", None)
    notes = cfg.wandb.get("notes", None)
    job_type = cfg.wandb.get("job_type", "wm_train")
    mode = str(cfg.wandb.get("mode", "online"))

    return wandb.init(
        entity=str(entity) if entity else None,
        project=str(cfg.wandb.project),
        name=str(name) if name else None,
        group=str(group) if group else None,
        tags=[str(t) for t in raw_tags] if raw_tags else None,
        notes=str(notes) if notes else None,
        job_type=str(job_type) if job_type else None,
        mode=mode,
        dir=str(out_dir.resolve()),
        config=wandb_cfg,
    )


@hydra.main(config_path="../configs", config_name="train_sanity_l2", version_base=None)
def main(cfg: DictConfig) -> None:
    rank, local_rank, world_size, device = _init_distributed()
    is_main = rank == 0
    torch.manual_seed(int(cfg.train.seed) + rank)

    out_dir = Path(cfg.train.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info("Training config:\n%s", OmegaConf.to_yaml(cfg))

    min_l1_frames = compute_min_l1_frames(list(cfg.strides), list(cfg.context_lengths))
    cfg_num_l1 = cfg.get("num_l1_frames", None)
    if cfg_num_l1 is not None:
        num_l1_frames = int(cfg_num_l1)
        if num_l1_frames < min_l1_frames:
            raise ValueError(
                f"num_l1_frames={num_l1_frames} is smaller than required minimum {min_l1_frames} "
                f"for strides={list(cfg.strides)} and context_lengths={list(cfg.context_lengths)}"
            )
    else:
        num_l1_frames = min_l1_frames

    sources = [OmegaConf.to_container(d, resolve=True) for d in cfg.datasets]
    train_ds = HierarchicalClipDataset(
        sources,
        frame_skip=int(cfg.frame_skip),
        num_l1_frames=num_l1_frames,
        action_dim=int(cfg.action_dim),
        split="train",
        val_ratio=float(cfg.val_split),
        seed=int(cfg.train.seed),
    )
    val_ds = HierarchicalClipDataset(
        sources,
        frame_skip=int(cfg.frame_skip),
        num_l1_frames=num_l1_frames,
        action_dim=int(cfg.action_dim),
        split="val",
        val_ratio=float(cfg.val_split),
        seed=int(cfg.train.seed),
    )

    collate_fn = make_ptv3_collate_fn(
        preset=str(cfg.backbone.preset),
        scale=float(cfg.backbone.scale),
        center=bool(cfg.backbone.center),
    )
    bs = int(cfg.train.batch_size_per_gpu)
    train_sampler = train_ds.make_distributed_weighted_sampler(
        num_replicas=world_size,
        rank=rank,
        seed=int(cfg.train.seed),
        drop_last=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=train_sampler,
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        collate_fn=collate_fn,
        drop_last=len(train_sampler) >= bs,
        pin_memory=(device.type == "cuda"),
    )
    # Diagnostics only consume ~diag_num_frames frames; draw that many clips at random (fixed seed)
    # so probes cover all sources instead of the first few episodes of the first store.
    diag_clips = max(1, math.ceil(int(cfg.train.diag_num_frames) / num_l1_frames))
    diag_gen = torch.Generator().manual_seed(int(cfg.train.seed))
    diag_idx = torch.randperm(len(val_ds), generator=diag_gen)[:diag_clips].tolist()
    val_loader = DataLoader(
        Subset(val_ds, diag_idx),
        batch_size=bs,
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model_from_cfg(cfg).to(device)
    if world_size > 1:
        ddp_kwargs = {"device_ids": [local_rank]} if device.type == "cuda" else {}
        model = DDP(model, find_unused_parameters=False, **ddp_kwargs)

    raw_model = model.module if isinstance(model, DDP) else model
    param_groups = raw_model.get_param_groups(
        lr_backbone=float(cfg.train.lr_backbone),
        lr_head=float(cfg.train.lr_head),
        weight_decay=float(cfg.train.weight_decay),
    )
    optimizer = torch.optim.AdamW(param_groups)

    epochs = int(cfg.train.epochs)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    if cfg.train.max_steps is not None:
        total_steps = min(total_steps, int(cfg.train.max_steps))
    warmup_steps = max(1, int(round(float(cfg.train.warmup_ratio) * total_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _cosine_lr_factor(s, total_steps, warmup_steps)
    )

    precision = str(cfg.train.precision).lower()
    use_amp = device.type == "cuda" and precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision == "fp16"))

    total_params = sum(p.numel() for p in raw_model.parameters())
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    runtime_meta: dict[str, Any] = {
        "world_size": world_size,
        "global_batch_size": bs * world_size,
        "num_l1_frames": num_l1_frames,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "train_clips": len(train_ds),
        "val_clips": len(val_ds),
        "total_params": total_params,
        "trainable_params": trainable_params,
    }

    wandb_run = None
    exit_code = 0
    try:
        wandb_run = init_wandb_run(
            cfg,
            is_main=is_main,
            out_dir=out_dir,
            runtime_meta=runtime_meta,
        )
        log_every = max(1, int(cfg.wandb.get("log_every", 10)))
        # Diagnostics + checkpoint every eval_every steps (null: only at epoch end).
        eval_every = int(cfg.train.get("eval_every", None) or 0)

        def evaluate_and_checkpoint(epoch: int, step: int) -> None:
            if not is_main:
                return
            diag = run_epoch_diagnostics(
                raw_model,
                val_loader,
                device=device,
                output_dir=out_dir,
                epoch=epoch,
                max_frames=int(cfg.train.diag_num_frames),
                ply_frames=int(cfg.train.diag_ply_frames),
                utonia_scale=float(cfg.backbone.scale),
            )
            log.info("epoch=%d step=%d diagnostics: %s", epoch, step, diag)
            if wandb_run is not None:
                wandb_run.log({"epoch": epoch, **diag}, step=step)

            ckpt = {
                "epoch": epoch,
                "global_step": step,
                "model_state_dict": raw_model.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            torch.save(ckpt, out_dir / "checkpoint_latest.pt")

        global_step = 0
        last_eval_step = -1
        for epoch in range(1, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            log_window_start = time.perf_counter()
            log_window_steps = 0

            for batch in train_loader:
                if cfg.train.max_steps is not None and global_step >= int(cfg.train.max_steps):
                    break
                B = int(batch["batch_size"])
                T = int(batch["seq_len"])
                point = move_point_to_device(batch["point"], device)
                actions = batch["actions"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                ctx = torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
                with ctx:
                    out = model(point, actions, batch_size=B, seq_len=T, return_attn=False)
                    loss = out["loss"]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in raw_model.parameters() if p.requires_grad],
                    float(cfg.train.grad_clip),
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1
                log_window_steps += 1

                if global_step == 1 or global_step % log_every == 0:
                    metric_keys = sorted(out["metrics"].keys())
                    metric_vals = torch.stack(
                        [
                            out["metrics"][k].detach().to(device=device, dtype=torch.float32)
                            if isinstance(out["metrics"][k], torch.Tensor)
                            else torch.tensor(float(out["metrics"][k]), device=device, dtype=torch.float32)
                            for k in metric_keys
                        ]
                    )
                    if world_size > 1 and dist.is_initialized():
                        dist.all_reduce(metric_vals, op=dist.ReduceOp.SUM)
                        metric_vals = metric_vals / float(world_size)

                    if is_main:
                        reduced_metrics = dict(zip(metric_keys, metric_vals.tolist()))
                        log_dict: dict[str, float | int] = {
                            f"train/{k}": float(v) for k, v in reduced_metrics.items()
                        }
                        for idx, group in enumerate(optimizer.param_groups):
                            group_name = group.get("name", idx)
                            log_dict[f"train/lr_{group_name}"] = float(group["lr"])
                        log_dict["train/grad_norm"] = float(grad_norm.detach().item())
                        elapsed = max(1e-6, time.perf_counter() - log_window_start)
                        step_time = elapsed / max(1, log_window_steps)
                        log_dict["train/step_time_s"] = step_time
                        log_dict["train/samples_per_sec"] = (bs * world_size) / step_time
                        log_dict["epoch"] = epoch
                        log_window_start = time.perf_counter()
                        log_window_steps = 0

                        log.info("step=%d epoch=%d loss=%.4f", global_step, epoch, log_dict["train/loss"])
                        if wandb_run is not None:
                            wandb_run.log(log_dict, step=global_step)

                if eval_every and global_step % eval_every == 0:
                    evaluate_and_checkpoint(epoch, global_step)
                    last_eval_step = global_step

            if last_eval_step != global_step:
                evaluate_and_checkpoint(epoch, global_step)
                last_eval_step = global_step

            if cfg.train.max_steps is not None and global_step >= int(cfg.train.max_steps):
                break
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
