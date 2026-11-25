"""Stage-1 DDP training driver for Point-H-JEPA (4x H100 or single-GPU fallback)."""

from __future__ import annotations

import logging
import math
import os
from contextlib import nullcontext
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=int(cfg.train.num_workers),
        collate_fn=collate_fn,
        drop_last=len(train_ds) >= bs,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
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

    wandb_run = None
    if is_main and bool(cfg.wandb.enabled):
        import wandb

        wandb_run = wandb.init(
            project=str(cfg.wandb.project),
            mode=str(cfg.wandb.mode),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    global_step = 0
    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()

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
            torch.nn.utils.clip_grad_norm_(
                [p for p in raw_model.parameters() if p.requires_grad],
                float(cfg.train.grad_clip),
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            if is_main and (global_step == 1 or global_step % 10 == 0):
                log_dict = {k: float(v.detach().item()) for k, v in out["metrics"].items()}
                log_dict["train/lr_head"] = optimizer.param_groups[-1]["lr"]
                log.info("step=%d epoch=%d loss=%.4f", global_step, epoch, log_dict["loss"])
                if wandb_run is not None:
                    wandb_run.log(log_dict, step=global_step)

        if is_main:
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
            log.info("epoch=%d diagnostics: %s", epoch, diag)
            if wandb_run is not None:
                wandb_run.log(diag, step=global_step)

            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": raw_model.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            torch.save(ckpt, out_dir / "checkpoint_latest.pt")

        if cfg.train.max_steps is not None and global_step >= int(cfg.train.max_steps):
            break

    if wandb_run is not None:
        wandb_run.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
