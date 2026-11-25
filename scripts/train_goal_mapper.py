"""Stage-2 Hindsight 3D Point Goal Mapper trainer (frozen world model)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from point_hjepa.data.clip_dataset import HindsightGoalDataset, make_goal_collate_fn
from point_hjepa.diagnostics.eval_probes import move_point_to_device
from point_hjepa.models.goal_mapper import HierarchicalGoalMapper
from scripts.train_ddp import build_model_from_cfg

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="train_sanity_l2", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg.train.seed))

    ckpt_path = Path(cfg.goal_mapper.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    wm_cfg = OmegaConf.create(ckpt["config"]) if "config" in ckpt else cfg
    world_model = build_model_from_cfg(wm_cfg)
    world_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    world_model.to(device).eval()
    for p in world_model.parameters():
        p.requires_grad = False

    sources = [OmegaConf.to_container(d, resolve=True) for d in cfg.datasets]
    train_ds = HindsightGoalDataset(
        sources,
        split="train",
        val_ratio=float(cfg.val_split),
        min_delta=int(cfg.goal_mapper.min_delta),
        max_delta=int(cfg.goal_mapper.max_delta),
        seed=int(cfg.train.seed),
    )
    val_ds = HindsightGoalDataset(
        sources,
        split="val",
        val_ratio=float(cfg.val_split),
        min_delta=int(cfg.goal_mapper.min_delta),
        max_delta=int(cfg.goal_mapper.max_delta),
        seed=int(cfg.train.seed),
    )

    collate_fn = make_goal_collate_fn(
        preset=str(wm_cfg.backbone.preset),
        scale=float(wm_cfg.backbone.scale),
        center=bool(wm_cfg.backbone.center),
    )
    bs = int(cfg.goal_mapper.batch_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        collate_fn=collate_fn,
        drop_last=len(train_ds) >= bs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        collate_fn=collate_fn,
        drop_last=False,
    )

    goal_mapper = HierarchicalGoalMapper(
        state_dim=int(wm_cfg.latent_dim),
        hidden_dim=int(cfg.goal_mapper.hidden_dim),
        fourier_bands=int(cfg.goal_mapper.fourier_bands),
    ).to(device)
    optimizer = torch.optim.AdamW(
        goal_mapper.parameters(),
        lr=float(cfg.goal_mapper.lr),
        weight_decay=float(cfg.goal_mapper.weight_decay),
    )

    out_dir = Path(cfg.goal_mapper.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(cfg.goal_mapper.epochs)
    max_steps = cfg.goal_mapper.max_steps
    global_step = 0

    for epoch in range(1, epochs + 1):
        goal_mapper.train()
        for batch in train_loader:
            if max_steps is not None and global_step >= int(max_steps):
                break
            pt_t = move_point_to_device(batch["point_t"], device)
            pt_tp = move_point_to_device(batch["point_tp"], device)
            p_g = batch["achieved_point_tp"].to(device)

            with torch.no_grad():
                z_t = world_model.encode_level1(pt_t)["z1"].float()
                z_tp = world_model.encode_level1(pt_tp)["z1"].float()

            optimizer.zero_grad(set_to_none=True)
            g_hat = goal_mapper(z_t, p_g)
            loss = F.mse_loss(g_hat, z_tp)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step == 1 or global_step % 20 == 0:
                log.info("goal_mapper step=%d epoch=%d mse=%.4f", global_step, epoch, float(loss.item()))

        goal_mapper.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                pt_t = move_point_to_device(batch["point_t"], device)
                pt_tp = move_point_to_device(batch["point_tp"], device)
                p_g = batch["achieved_point_tp"].to(device)
                z_t = world_model.encode_level1(pt_t)["z1"].float()
                z_tp = world_model.encode_level1(pt_tp)["z1"].float()
                g_hat = goal_mapper(z_t, p_g)
                val_losses.append(float(F.mse_loss(g_hat, z_tp).item()))
                if max_steps is not None and len(val_losses) >= 2:
                    break
        val_mse = sum(val_losses) / max(1, len(val_losses))
        log.info("epoch=%d goal_mapper val_mse=%.4f", epoch, val_mse)

        torch.save(
            {
                "epoch": epoch,
                "goal_mapper_state_dict": goal_mapper.state_dict(),
                "val_mse": val_mse,
            },
            out_dir / "goal_mapper_latest.pt",
        )
        if max_steps is not None and global_step >= int(max_steps):
            break


if __name__ == "__main__":
    main()
