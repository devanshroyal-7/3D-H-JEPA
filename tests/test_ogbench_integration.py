"""Unit and integration tests for OGBench data collection, eval harness, and baselines."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from point_hjepa.data.clip_dataset import HierarchicalClipDataset, make_ptv3_collate_fn
from point_hjepa.data.collect_ogbench_6d import OGBENCH_TASKS, backproject_depth_to_world
from point_hjepa.data.zarr_writer import ZarrEpisodeWriter
from point_hjepa.eval.baselines import TAB1_OGBENCH_REFERENCE, format_comparison_table
from point_hjepa.eval.encoders import PointHJEPAEncoder, StateOnlyEncoder
from point_hjepa.eval.jepa4r_adapter import adapter_dim, encode_pcd, load_frozen
from point_hjepa.eval.ogbench_policies import (
    FLAREBCConfig,
    GCIQLConfig,
    GCIQLGoalSampler,
    TorchFLAREBCPolicy,
    TorchGCIQLPolicy,
    build_flare_features,
    train_flare_bc,
)
from point_hjepa.models.point_hjepa import PointHJEPA


def test_backproject_unsegmented_keeps_background_drops_invisible() -> None:
    H, W = 16, 16
    depth = np.full((H, W), 1.0, dtype=np.float32)
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    geom_ids = np.full((H, W), 0, dtype=np.int32)  # 0 = table/background

    # Background color: [10, 20, 30]
    rgb[:, :] = np.array([10, 20, 30], dtype=np.uint8)
    # Robot arm region: geom 1, color [0, 128, 255]
    geom_ids[4:10, 4:8] = 1
    rgb[4:10, 4:8] = np.array([0, 128, 255], dtype=np.uint8)
    # Object region: geom 2, color [200, 50, 50]
    geom_ids[10:14, 6:12] = 2
    rgb[10:14, 6:12] = np.array([200, 50, 50], dtype=np.uint8)
    # Invisible mocap goal marker: geom 99, color [255, 0, 255]
    geom_ids[1:3, 1:4] = 99
    rgb[1:3, 1:4] = np.array([255, 0, 255], dtype=np.uint8)
    # Sky pixels: geom -1, invalid depth
    geom_ids[0, :] = -1
    depth[0, :] = 100.0

    cam_xpos = np.array([0.5, 0.0, 1.0], dtype=np.float64)
    cam_xmat = np.eye(3, dtype=np.float64)
    rng = np.random.default_rng(7)

    # Unsegmented keeps table/background [10, 20, 30] and drops invisible mocap [255, 0, 255]
    xyz_u, rgb_u = backproject_depth_to_world(
        depth,
        rgb,
        fovy_deg=45.0,
        cam_xpos=cam_xpos,
        cam_xmat=cam_xmat,
        geom_ids=geom_ids,
        drop_geom_ids={99},
        background_geom_ids={0},
        object_geom_ids={2},
        unsegmented=True,
        voxel_size=None,
        num_points=256,
        sort_by_voxel=True,
        rng=rng,
    )
    assert xyz_u.shape == (256, 3) and xyz_u.dtype == np.float32
    assert rgb_u.shape == (256, 3) and rgb_u.dtype == np.uint8
    assert np.isfinite(xyz_u).all()

    has_bg = np.any(np.all(rgb_u == np.array([10, 20, 30], dtype=np.uint8), axis=-1))
    has_marker = np.any(np.all(rgb_u == np.array([255, 0, 255], dtype=np.uint8), axis=-1))
    assert has_bg, "Unsegmented mode must retain physical background/tabletop geometry"
    assert not has_marker, "Unsegmented mode must still exclude invisible mocap goal markers"

    # Segmented mode drops background AND invisible marker, and stratifies 25% objects
    xyz_s, rgb_s = backproject_depth_to_world(
        depth,
        rgb,
        fovy_deg=45.0,
        cam_xpos=cam_xpos,
        cam_xmat=cam_xmat,
        geom_ids=geom_ids,
        drop_geom_ids={99},
        background_geom_ids={0},
        object_geom_ids={2},
        unsegmented=False,
        voxel_size=None,
        num_points=100,
        object_fraction=0.25,
        sort_by_voxel=False,
        rng=rng,
    )
    assert xyz_s.shape == (100, 3)
    assert not np.any(np.all(rgb_s == np.array([10, 20, 30], dtype=np.uint8), axis=-1))
    assert not np.any(np.all(rgb_s == np.array([255, 0, 255], dtype=np.uint8), axis=-1))
    n_obj = int(np.sum(np.all(rgb_s == np.array([200, 50, 50], dtype=np.uint8), axis=-1)))
    assert n_obj == 24 or n_obj == 25


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for Triton PTv3 kernels")
def test_ogbench_zarr_loads_into_clip_dataset_and_point_hjepa(tmp_path: Path) -> None:
    zarr_path = tmp_path / "ogbench_cube_single_unseg.zarr"
    writer = ZarrEpisodeWriter(
        zarr_path,
        num_points=256,
        action_dim=5,
        proprio_dim=20,
        xyz_dtype="float16",
        attrs={"task": "cube_single", "unsegmented": True},
    )
    rng = np.random.default_rng(11)
    for _ in range(4):
        T = 40
        writer.append_episode(
            xyz=rng.uniform(0.1, 0.7, size=(T, 256, 3)).astype(np.float32),
            rgb=rng.integers(0, 256, size=(T, 256, 3), dtype=np.uint8),
            action=rng.uniform(-1.0, 1.0, size=(T, 5)).astype(np.float32),
            proprio=rng.normal(size=(T, 20)).astype(np.float32),
            achieved_point=rng.uniform(0.2, 0.6, size=(T, 3)).astype(np.float32),
            ee_point=rng.uniform(0.2, 0.6, size=(T, 3)).astype(np.float32),
        )

    ds = HierarchicalClipDataset(zarr_path, frame_skip=5, num_l1_frames=7, action_dim=5, split="train", val_ratio=0.25)
    collate_fn = make_ptv3_collate_fn(preset="utonia_tiny", scale=4.0, center=False)
    batch = collate_fn([ds[0], ds[1]])
    assert batch["actions"].shape == (2, 6, 25)

    device = torch.device("cuda")
    for k in ("coord", "grid_coord", "feat", "offset"):
        batch["point"][k] = batch["point"][k].to(device)
    actions = batch["actions"].to(device)

    model = PointHJEPA(
        num_levels=2,
        strides=(2,),
        context_lengths=(3, 3),
        action_dim=5,
        frame_skip=5,
        latent_dim=64,
        action_l1_dim=32,
        backbone_pretrained=False,
        sigreg_num_slices=64,
    ).to(device)
    out = model(batch["point"], actions, batch_size=2, seq_len=7)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters() if p.requires_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for Triton PTv3 kernels")
def test_point_hjepa_frozen_encoder_modes(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "num_levels": 2,
            "strides": [2],
            "context_lengths": [3, 3],
            "action_dim": 5,
            "frame_skip": 5,
            "latent_dim": 64,
            "action_l1_dim": 32,
            "macro_action_dim": 8,
            "action_pooler_type": "transformer",
            "backbone": {
                "preset": "utonia_tiny",
                "pretrained": False,
                "scale": 4.0,
                "center": False,
                "serialization_depth": 12,
                "freeze_stages": 3,
                "fourier_bands": 8,
            },
            "pooler": {"num_queries": 4, "num_heads": 4, "num_layers": 2, "mlp_ratio": 4.0},
            "predictor": {"depth": 2, "num_heads": 4, "mlp_ratio": 2.0},
            "loss": {
                "lambda_sigreg": [1.28, 2.56],
                "gamma_idm": [50.0, 50.0],
                "sigreg_num_slices": 64,
                "sigreg_t_min": 0.2,
                "sigreg_t_max": 4.0,
                "sigreg_num_knots": 16,
            },
        }
    )
    wm = PointHJEPA(
        num_levels=2,
        strides=(2,),
        context_lengths=(3, 3),
        action_dim=5,
        frame_skip=5,
        latent_dim=64,
        action_l1_dim=32,
        backbone_pretrained=False,
        predictor_depth=2,
        predictor_heads=4,
        predictor_mlp_ratio=2.0,
        sigreg_num_slices=64,
        sigreg_num_knots=16,
    )
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": wm.state_dict(), "config": OmegaConf.to_container(cfg, resolve=True)}, ckpt_path)

    clouds = np.concatenate(
        [
            np.random.uniform(0.1, 0.7, size=(3, 256, 3)).astype(np.float32),
            np.random.randint(0, 256, size=(3, 256, 3)).astype(np.float32),
        ],
        axis=-1,
    )
    for mode, exp_dim in (("l1", 64), ("l2", 64), ("concat", 128)):
        enc = PointHJEPAEncoder(ckpt_path, device="cuda", latent_mode=mode, batch_size=2)
        assert enc.latent_dim == exp_dim
        assert not enc.model.training
        assert all(not p.requires_grad for p in enc.model.parameters())
        feats = enc.encode_batch(clouds)
        assert feats.shape == (3, exp_dim)
        assert feats.dtype == np.float32
        assert np.isfinite(feats).all()

    assert adapter_dim(ckpt_path, "concat") == 128
    ad_enc = load_frozen(ckpt_path, device="cuda", latent_mode="concat")
    single_feat = encode_pcd(ad_enc, clouds[0])
    assert single_feat.shape == (128,) and np.isfinite(single_feat).all()


def test_torch_gciql_sampler_and_update_step() -> None:
    rng = np.random.default_rng(3)
    for num_cubes, lat_dim in ((1, 256), (2, 512)):
        n_eps, ep_len = 4, 15
        total = n_eps * ep_len
        state_dim = 19 + 9 * num_cubes
        states = rng.normal(size=(total, state_dim)).astype(np.float32)
        actions = rng.uniform(-1.0, 1.0, size=(total, 5)).astype(np.float32)
        feats = rng.normal(size=(total, lat_dim)).astype(np.float32)
        ep_ends = np.arange(1, n_eps + 1, dtype=np.int64) * ep_len

        cfg = GCIQLConfig(obs_dim=lat_dim + 19, goal_dim=lat_dim + 3 * num_cubes, action_dim=5, batch_size=16)
        sampler = GCIQLGoalSampler(states, actions, ep_ends, num_cubes=num_cubes, features=feats, cfg=cfg, seed=5)
        batch = sampler.sample(16)
        assert batch["observations"].shape == (16, lat_dim + 19)
        assert batch["actor_goals"].shape == (16, lat_dim + 3 * num_cubes)
        assert set(np.unique(batch["rewards"]).tolist()).issubset({-1.0, 0.0})

        policy = TorchGCIQLPolicy(cfg)
        q1_before = policy.target_q1[0].weight.detach().clone()
        metrics = policy.update(batch, device="cpu")
        assert all(np.isfinite(v) for v in metrics.values())
        assert not torch.allclose(q1_before, policy.target_q1[0].weight)

        act = policy.act(batch["observations"][0], batch["actor_goals"][0], device="cpu")
        assert act.shape == (5,) and np.all(act >= -1.0) and np.all(act <= 1.0)

    state_enc = StateOnlyEncoder(28)
    assert state_enc.encode_one(np.ones(28, dtype=np.float32)).shape == (28,)


def test_torch_flare_bc_train_step() -> None:
    rng = np.random.default_rng(19)
    T, D = 60, 32
    latents = rng.normal(size=(T, D)).astype(np.float32)
    proprio = rng.normal(size=(T, 8)).astype(np.float32)
    feats = build_flare_features(latents, proprio, frame_stack=3, proprio_dim=6)
    assert feats.shape == (T, 3 * D + 6)
    # Boundary t=0 uses z_0 - z_0 = 0 differences
    assert np.allclose(feats[0, D : 3 * D], 0.0)

    actions = np.tanh(feats[:, :5]).astype(np.float32)
    cfg = FLAREBCConfig(latent_dim=D, proprio_dim=6, action_dim=5, batch_size=16, epochs=4)
    policy, _ = train_flare_bc(feats, actions, cfg, device="cpu")
    act = policy.act(feats[0], device="cpu")
    assert act.shape == (5,) and np.all(act >= -1.0) and np.all(act <= 1.0)


def test_eval_ogbench_cli_print_baselines_and_dry_run() -> None:
    res1 = subprocess.run(
        [sys.executable, "scripts/eval_ogbench.py", "--print-baselines"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "78.4" in res1.stdout and "64.8" in res1.stdout
    assert "ogbench-cube-single" in res1.stdout

    res2 = subprocess.run(
        [
            sys.executable,
            "scripts/eval_ogbench.py",
            "--dry-run",
            "--methods",
            "point_hjepa",
            "jepa4r",
            "lewm",
            "state",
            "--tasks",
            "cube_single",
            "scene",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "point_hjepa_concat_unseg_4096" in res2.stdout
    assert "paper_ref=78.4%" in res2.stdout

    md = format_comparison_table({"point_hjepa": {"cube_single": 80.0, "scene": 68.0}}, ["cube_single", "scene"])
    assert "| point_hjepa | 80.0 | 68.0 | 74.0 | +2.4 |" in md
    assert TAB1_OGBENCH_REFERENCE["jepa4r"]["cube_single"] == 78.4
    assert len(OGBENCH_TASKS) == 8


def test_setup_jepa4r_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hydra import compose, initialize_config_dir

    repo_root = Path(__file__).resolve().parent.parent
    monkeypatch.syspath_prepend(str(repo_root))
    from scripts import setup_jepa4r

    config_dir = str(repo_root / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        wm_cfg = compose(config_name="train_ogbench_v1")
        col_cfg = compose(config_name="collect_ogbench")
    assert wm_cfg.action_dim == 5
    assert wm_cfg.frame_skip == 5
    assert list(wm_cfg.loss.lambda_sigreg) == [1.28, 2.56]
    assert col_cfg.unsegmented is True
    assert col_cfg.num_points == 4096

    assert setup_jepa4r.resolve_branch("cube-single") == "ogbench-cube-single"
    assert setup_jepa4r.resolve_branch("scene") == "ogbench-scene"
    with pytest.raises(ValueError):
        setup_jepa4r.resolve_branch("unknown_task")

    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], check: bool = False) -> None:
        calls.append(list(cmd))

    monkeypatch.setattr(setup_jepa4r.subprocess, "run", _fake_run)
    monkeypatch.setattr(setup_jepa4r.subprocess, "check_output", lambda *a, **k: "abc1234\n")
    dest = setup_jepa4r.setup_code(tmp_path / "JEPA4R", commit="deadbeef")
    assert dest == tmp_path / "JEPA4R"
    assert "clone" in calls[0]
    assert "checkout" in calls[1] and "deadbeef" in calls[1]

    dl_calls: list[dict[str, str]] = []
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kw: dl_calls.append(kw) or kw["local_dir"],
    )
    out = setup_jepa4r.setup_weights(tmp_path / "ckpts", ["cube_single", "scene"])
    assert len(out) == 2
    assert dl_calls[0]["revision"] == "ogbench-cube-single"
    assert dl_calls[1]["revision"] == "ogbench-scene"
