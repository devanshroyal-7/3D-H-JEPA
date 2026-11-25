"""Unit and integration verification tests for Point-H-JEPA."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import point_transformer_v3 as ptv3
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import trimesh
import zarr

from point_hjepa.diagnostics.eval_probes import compute_effective_rank, fit_and_eval_coord_probe
from point_hjepa.diagnostics.export_attn_ply import export_attention_ply
from point_hjepa.models.attentive_pooler import AttentivePoolerBridge
from point_hjepa.models.goal_mapper import HierarchicalGoalMapper
from point_hjepa.models.point_hjepa import PointHJEPA
from point_hjepa.models.sigreg import SIGReg
from point_hjepa.models.utonia_backbone import UtoniaBackbone


def _make_collated_clouds(
    num_clouds: int,
    points_per_cloud: list[int] | None = None,
    device: torch.device = torch.device("cpu"),
):
    pts = []
    for i in range(num_clouds):
        n = 512 if points_per_cloud is None else points_per_cloud[i]
        coord = torch.rand(n, 3) * 0.6 + torch.tensor([0.1, -0.3, 0.0])
        color = torch.rand(n, 3) * 255.0
        normal = torch.zeros(n, 3)
        pt, _ = ptv3.prepare(
            coord, color, normal, preset="utonia_tiny", scale=4.0, center=False, device="cpu"
        )
        pts.append(pt)
    batch = ptv3.collate(pts)
    for k in ("coord", "grid_coord", "feat", "offset"):
        batch[k] = batch[k].to(device)
    return batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for Triton PTv3 kernels")
def test_utonia_backbone_freeze_and_grad() -> None:
    device = torch.device("cuda")
    backbone = UtoniaBackbone(
        preset="utonia_tiny",
        pretrained=True,
        serialization_depth=12,
        freeze_stages=3,
        out_dim=256,
        fourier_bands=8,
    ).to(device)
    backbone.train()

    # Frozen stem + enc[0..2] must stay in eval mode with requires_grad=False
    assert not backbone.backbone.embedding.training
    for idx in range(3):
        assert not backbone.backbone.enc[idx].training
        for p in backbone.backbone.enc[idx].parameters():
            assert not p.requires_grad
    for idx in range(3, 5):
        assert backbone.backbone.enc[idx].training
        for p in backbone.backbone.enc[idx].parameters():
            assert p.requires_grad

    batch = _make_collated_clouds(4, [1024, 1536, 800, 1200], device=device)
    tokens, coord, offset = backbone(batch)
    assert tokens.ndim == 2 and tokens.shape[1] == 256
    assert coord.shape[0] == tokens.shape[0]
    assert offset.tolist()[-1] == tokens.shape[0]

    loss = tokens.float().pow(2).mean()
    loss.backward()

    for p in backbone.backbone.embedding.parameters():
        assert p.grad is None
    for idx in range(3):
        for p in backbone.backbone.enc[idx].parameters():
            assert p.grad is None
    stage4_grads = [p.grad for p in backbone.backbone.enc[3].parameters() if p.grad is not None]
    assert len(stage4_grads) > 0 and any(g.abs().sum().item() > 0 for g in stage4_grads)


def test_attentive_pooler_ragged_mask_and_attn() -> None:
    pooler = AttentivePoolerBridge(dim=256, num_queries=4, num_heads=4, num_layers=2)
    tokens = torch.randn(25, 256, requires_grad=True)
    offset = torch.tensor([10, 25])  # Frame 0 has 10 tokens, Frame 1 has 15 tokens

    z1, attn, lengths = pooler(tokens, offset)
    assert z1.shape == (2, 256)
    assert attn.shape == (2, 15)
    assert lengths.tolist() == [10, 15]
    # Padded positions of frame 0 (indices 10..14) must receive zero attention weight.
    assert torch.allclose(attn[0, 10:], torch.zeros(5), atol=1e-7)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2), atol=1e-5)

    (z1 ** 2).sum().backward()
    assert tokens.grad is not None and tokens.grad.abs().sum().item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for Triton PTv3 kernels")
def test_hjepa_shapes_and_idm_stop_grad() -> None:
    device = torch.device("cuda")
    B, T = 2, 8  # (c_2 + 1) * s_2 = (3 + 1) * 2 = 8
    model = PointHJEPA(
        num_levels=2,
        strides=(2,),
        context_lengths=(3, 3),
        action_dim=8,
        frame_skip=3,
        backbone_pretrained=False,
        sigreg_num_slices=128,
    ).to(device)
    model.train()

    batch = _make_collated_clouds(B * T, [400 + 20 * i for i in range(B * T)], device=device)
    actions_u1 = torch.randn(B, T - 1, 24, device=device)

    out = model(batch, actions_u1, batch_size=B, seq_len=T, return_attn=True)
    assert out["states"][0].shape == (B, 8, 256)
    assert out["states"][1].shape == (B, 4, 256)
    assert out["actions"][0].shape == (B, 7, 64)
    assert out["actions"][1].shape == (B, 3, 8)

    # 1. Verify mandatory stop-grad on a^(2) in L_idm^(2): action_poolers[0] MUST get ZERO grad!
    model.zero_grad(set_to_none=True)
    out["metrics"]["l2/loss_idm"].backward(retain_graph=True)
    for p in model.action_poolers[0].parameters():
        assert p.grad is None or p.grad.abs().sum().item() == 0.0

    # 2. Verify full loss trains ALL unfrozen parameters (DDP find_unused_parameters=False safe)
    model.zero_grad(set_to_none=True)
    out = model(batch, actions_u1, batch_size=B, seq_len=T, return_attn=False)
    out["loss"].backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Unfrozen param {name} received no gradient!"


def _ddp_sigreg_worker(rank: int, world_size: int, z_full: torch.Tensor, ref_grad: torch.Tensor, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.manual_seed(9999)
    sigreg = SIGReg(num_slices=256, t_min=0.2, t_max=4.0, num_knots=32)

    chunk = z_full.shape[0] // world_size
    z_local = z_full[rank * chunk : (rank + 1) * chunk].clone().detach().requires_grad_(True)
    loss = sigreg(z_local)
    loss.backward()

    # torch.distributed.nn.functional.all_gather sums gradients across ranks so DDP's
    # 1/world_size average recovers the exact global-batch gradient.
    assert torch.allclose(z_local.grad / world_size, ref_grad[rank * chunk : (rank + 1) * chunk], atol=1e-5)
    dist.destroy_process_group()


def test_sigreg_single_and_ddp_all_gather() -> None:
    torch.manual_seed(1234)
    z_full = torch.randn(16, 4, 32, requires_grad=True)
    torch.manual_seed(9999)
    sigreg = SIGReg(num_slices=256, t_min=0.2, t_max=4.0, num_knots=32)
    loss = sigreg(z_full)
    loss.backward()
    ref_grad = z_full.grad.detach().clone()
    assert ref_grad.abs().sum().item() > 0

    port = 29611 + (os.getpid() % 500)
    mp.spawn(
        _ddp_sigreg_worker,
        args=(2, z_full.detach(), ref_grad, port),
        nprocs=2,
        join=True,
    )


def test_goal_mapper_and_propagation() -> None:
    mapper = HierarchicalGoalMapper(state_dim=256, hidden_dim=512, fourier_bands=6)
    z_t1 = torch.randn(8, 256, requires_grad=True)
    z_tp1 = torch.randn(8, 256)
    p_g = torch.randn(8, 3)

    g_hat1 = mapper(z_t1, p_g)
    assert g_hat1.shape == (8, 256)
    loss = torch.nn.functional.mse_loss(g_hat1, z_tp1)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in mapper.parameters())

    upper_enc = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.LayerNorm(256))
    goals = mapper.propagate_goals(z_t1.detach(), p_g, [upper_enc])
    assert len(goals) == 2 and goals[0].shape == (8, 256) and goals[1].shape == (8, 256)


def test_diagnostics_and_ply_export(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # Rank collapse triggers warning
    collapsed = torch.ones(64, 32)
    with caplog.at_level(logging.WARNING):
        eff_low, healthy_low = compute_effective_rank(collapsed)
    assert not healthy_low and eff_low < 16.0

    # Isotropic Gaussian has high rank
    iso = torch.randn(256, 32)
    eff_hi, healthy_hi = compute_effective_rank(iso)
    assert healthy_hi and eff_hi > 16.0

    # Coordinate probe on linear signal achieves < 2.0 cm error
    y = torch.rand(200, 3) * 0.5
    z = torch.cat([y, torch.randn(200, 13) * 0.01], dim=-1)
    err_cm = fit_and_eval_coord_probe(z[:160], y[:160], z[160:], y[160:], steps=250)
    assert err_cm < 2.0

    ply_path = export_attention_ply(
        torch.rand(128, 3) * 4.0,
        torch.rand(128),
        tmp_path / "attn.ply",
        utonia_scale=4.0,
    )
    loaded = trimesh.load(str(ply_path))
    assert loaded.vertices.shape == (128, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for end-to-end training smoke test")
def test_train_ddp_and_goal_mapper_smoke(tmp_path: Path) -> None:
    zarr_path = Path("outputs/val_stackcube/StackCube-v1_branched_exploratory.zarr")
    if not zarr_path.exists():
        # Fallback compact synthetic Zarr v3 store
        zarr_path = tmp_path / "synth.zarr"
        codec = zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
        root = zarr.open_group(str(zarr_path), mode="w")
        root.create_array("xyz", data=np.random.uniform(-0.1, 0.5, (80, 512, 3)).astype(np.float32), compressors=codec)
        root.create_array("rgb", data=np.random.randint(0, 256, (80, 512, 3), dtype=np.uint8), compressors=codec)
        root.create_array("action", data=np.random.uniform(-1, 1, (80, 8)).astype(np.float32), compressors=codec)
        root.create_array("proprio", data=np.zeros((80, 26), dtype=np.float32), compressors=codec)
        root.create_array("achieved_point", data=np.random.uniform(0.2, 0.7, (80, 3)).astype(np.float32), compressors=codec)
        root.create_array("ee_point", data=np.random.uniform(0.2, 0.7, (80, 3)).astype(np.float32), compressors=codec)
        root.create_array("episode_ends", data=np.array([40, 80], dtype=np.int64), compressors=codec)

    run_dir = tmp_path / "run_l2"
    gm_dir = tmp_path / "run_gm"
    ds_override = f"datasets=[{{path:{zarr_path},weight:1.0}}]"

    cmd_stage1 = [
        sys.executable,
        "scripts/train_ddp.py",
        ds_override,
        "train.batch_size_per_gpu=2",
        "train.epochs=1",
        "train.max_steps=2",
        "train.num_workers=0",
        "train.diag_num_frames=32",
        "train.diag_ply_frames=2",
        f"train.output_dir={run_dir}",
        "wandb.enabled=false",
    ]
    res1 = subprocess.run(cmd_stage1, check=True, capture_output=True, text=True)
    ckpt_file = run_dir / "checkpoint_latest.pt"
    assert ckpt_file.is_file(), res1.stderr
    assert len(list((run_dir / "attn_ply").glob("*.ply"))) == 2

    cmd_stage2 = [
        sys.executable,
        "scripts/train_goal_mapper.py",
        ds_override,
        f"goal_mapper.checkpoint={ckpt_file}",
        f"goal_mapper.output_dir={gm_dir}",
        "goal_mapper.batch_size=4",
        "goal_mapper.epochs=1",
        "goal_mapper.max_steps=2",
        "train.num_workers=0",
        "wandb.enabled=false",
    ]
    res2 = subprocess.run(cmd_stage2, check=True, capture_output=True, text=True)
    assert (gm_dir / "goal_mapper_latest.pt").is_file(), res2.stderr
