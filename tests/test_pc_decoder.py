"""Tests for the per-level JEPA4R point-cloud decoders (stub trunk; the real PAGE-4D trunk is opt-in)."""

from __future__ import annotations

import math
import os

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from point_hjepa.models.pc_decoder.hierarchical_decoder import HierarchicalPointDecoder
from point_hjepa.models.pc_decoder.lora import LevelRouter, LoRAQV, LoRAQVBank
from point_hjepa.models.pc_decoder.losses import chamfer, nn_distance, sliced_wasserstein
from point_hjepa.models.pc_decoder.page4d_trunk import DEFAULT_PAGE4D_CKPT, DEFAULT_PAGE4D_ROOT
from point_hjepa.models.pc_decoder.set_generator import SetDecoder


class _Attn(nn.Module):
    def __init__(self, dim: int, heads: int = 2) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        return x + self.proj(o)


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = _Attn(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)


class StubTrunk(nn.Module):
    """Same contract as ``PAGE4DTrunk`` (tokens -> raw 4-channel map + head features), tiny and random."""

    def __init__(self, embed_dim: int = 32, grid: int = 4, patch_size: int = 2, feat_dim: int = 8) -> None:
        super().__init__()
        self.grid, self.patch_size, self.embed_dim, self.feat_dim = grid, patch_size, embed_dim, feat_dim
        self.side = grid * patch_size
        self.blocks_a = nn.ModuleList([_Block(embed_dim) for _ in range(2)])
        self.blocks_b = nn.ModuleList([_Block(embed_dim)])
        self.to_feat = nn.Conv2d(embed_dim, feat_dim, 1)
        self.output_conv2 = nn.Conv2d(feat_dim, 4, 1)
        self.requires_grad_(False)

    def lora_block_lists(self) -> list[nn.Module]:
        return [self.blocks_a, self.blocks_b]

    def tokens_to_map(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = tokens
        for blk in [*self.blocks_a, *self.blocks_b]:
            x = blk(x)
        B, G = x.shape[0], self.grid
        f = self.to_feat(x.transpose(1, 2).reshape(B, -1, G, G))
        feats = F.interpolate(f, size=(self.side, self.side), mode="bilinear", align_corners=True).float()
        return self.output_conv2(feats), feats


def _decoder(num_levels: int = 2, latent_dim: int = 16) -> HierarchicalPointDecoder:
    torch.manual_seed(0)
    return HierarchicalPointDecoder(
        StubTrunk(),
        num_levels=num_levels,
        latent_dim=latent_dim,
        gen_dim=32,
        gen_hidden=64,
        gen_depth=2,
        query_dim=8,
        target_center=(0.5, 0.0, 0.25),
        target_scale=4.0,
    )


def test_set_decoder_shapes_and_zero_film_init() -> None:
    gen = SetDecoder(latent_dim=16, num_queries=9, out_dim=24, dim=32, hidden=64, depth=3, query_dim=8)
    z = torch.randn(5, 16)
    out = gen(z)
    assert out.shape == (5, 9, 24) and out.dtype == torch.float32
    # zero-initialised FiLM: modulation is identity, so the output cannot depend on the latent yet
    assert torch.allclose(out[0], out[1], atol=1e-6)
    gen.film.weight.data.normal_()
    out2 = gen(z)
    assert not torch.allclose(out2[0], out2[1], atol=1e-4)


def test_lora_bank_identity_at_init_and_per_sample_routing() -> None:
    torch.manual_seed(0)
    base = nn.Linear(12, 36)
    x = torch.randn(4, 5, 12)
    ref = base(x)

    single = LoRAQV(base, rank=2)
    assert torch.allclose(single(x), ref, atol=1e-6)

    router = LevelRouter()
    bank = LoRAQVBank(base, rank=2, num_banks=3, router=router)
    router.idx = torch.tensor([0, 1, 2, 1])
    assert torch.allclose(bank(x), ref, atol=1e-6)

    with torch.no_grad():
        bank.up_q[1].normal_()
        bank.up_v[1].normal_()
    out = bank(x)
    for b, lvl in enumerate(router.idx.tolist()):
        if lvl == 1:
            assert not torch.allclose(out[b, :, :12], ref[b, :, :12])  # q changed
            assert torch.allclose(out[b, :, 12:24], ref[b, :, 12:24], atol=1e-6)  # k untouched
            assert not torch.allclose(out[b, :, 24:], ref[b, :, 24:])  # v changed
        else:
            assert torch.allclose(out[b], ref[b], atol=1e-6)

    # bank b equals a single LoRAQV with the same weights
    single = LoRAQV(base, rank=2)
    with torch.no_grad():
        single.down.weight.copy_(bank.down[1])
        single.up_q.weight.copy_(bank.up_q[1])
        single.up_v.weight.copy_(bank.up_v[1])
    assert torch.allclose(out[1], single(x[1:2])[0], atol=1e-5)


def test_losses_known_values() -> None:
    pred = torch.tensor([[[0.0, 0.0, 0.0, 0.2, 0.2, 0.2]]])
    gt = torch.tensor([[[1.0, 0.0, 0.0, 0.5, 0.5, 0.5]]])
    loss, logs = chamfer(pred, gt, color_weight=0.1)
    assert math.isclose(float(logs["cd"]), 2.0, rel_tol=1e-6)  # 1 (pred->gt) + 1 (gt->pred), squared
    assert math.isclose(float(logs["col"]), 0.6, rel_tol=1e-5)  # 0.3 L1 each way
    assert math.isclose(float(loss), 2.06, rel_tol=1e-5)
    assert math.isclose(float(nn_distance(pred, gt)[0]), 1.0, rel_tol=1e-6)

    torch.manual_seed(0)
    cloud = torch.randn(2, 256, 3)
    assert float(chamfer(cloud, cloud, 0.0)[0]) == 0.0
    assert float(sliced_wasserstein(cloud, cloud, 64)) < 1e-10
    shift = torch.tensor([0.3, -0.2, 0.1])
    sw = float(sliced_wasserstein(cloud + shift, cloud, 4000))
    assert math.isclose(sw, float(shift.pow(2).sum()), rel_tol=0.1)  # x3 scaling -> |d|^2 in expectation


def test_frame_init_maps_optical_axis_to_target_origin() -> None:
    dec = _decoder()
    head = dec.heads[0]
    # PAGE-4D camera frame at unit scene scale: the look-at point is (0, 0, 1)
    out = head.frame(torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.allclose(out, torch.zeros(1, 3), atol=1e-5)
    world = dec.to_world(torch.zeros(1, 3))
    assert torch.allclose(world, torch.tensor([[0.5, 0.0, 0.25]]))
    xyz = torch.rand(2, 7, 3)
    rgb = torch.randint(0, 256, (2, 7, 3)).float()
    back = dec.to_world(dec.to_target(xyz, rgb))
    assert torch.allclose(back[..., :3], xyz, atol=1e-6) and torch.allclose(back[..., 3:], rgb, atol=1e-4)


def test_hierarchical_decoder_shapes_and_level_isolation() -> None:
    dec = _decoder(num_levels=3)
    dec.train()
    z = torch.randn(6, 16)
    levels = torch.tensor([0, 2, 0, 2, 2, 0])
    dense = dec(z, levels)
    side = dec.trunk.side
    assert dense.shape == (6, side * side, 6)
    assert dense[..., 3:].min() >= 0 and dense[..., 3:].max() <= 1

    # a sample decoded alone equals the same sample in the mixed batch
    alone = dec(z[1:2], 2)
    assert torch.allclose(alone[0], dense[1], atol=1e-5)

    target = torch.rand(6, 20, 6)
    loss, logs = dec.loss(dense, target, num_points=20)
    assert {"cd", "col", "sw"} <= set(logs)
    loss.backward()
    for l, head in enumerate(dec.heads):
        grads = [p.grad for p in head.parameters()]
        if l == 1:  # absent from the batch
            assert all(g is None for g in grads)
        else:
            assert any(g is not None and g.abs().sum() > 0 for g in grads)
    for p in dec.trunk.parameters():
        assert p.grad is None

    pts = dec.decode_points(z[:2], 0, num_points=50)
    assert pts.shape == (2, 50, 6)
    assert torch.equal(dec.pick(dense, 10, fixed=True), dec.pick(dense, 10, fixed=True))


def test_lora_stage_b_grads_and_checkpoint_roundtrip(tmp_path) -> None:
    dec = _decoder(num_levels=2)
    stage_a = dec.trainable_state_dict()
    assert not any(k.startswith("trunk.") for k in stage_a)

    adapters = dec.enable_lora(rank=2)
    assert len(adapters) == 3  # one per fused qkv of the stub's block lists
    z = torch.randn(4, 16)
    dense = dec(z, torch.tensor([1, 1, 1, 1]))
    dec.loss(dense, torch.rand(4, 16, 6), num_points=16)[0].backward()
    # up-projections start at zero, so only they (and only bank 1) receive gradient at first
    for a in adapters:
        assert a.up_q.grad[1].abs().sum() > 0 and a.up_q.grad[0].abs().sum() == 0

    sd = dec.trainable_state_dict()
    assert any(k.endswith(".up_q") for k in sd) and all(
        not k.startswith("trunk.") or k.endswith((".down", ".up_q", ".up_v")) for k in sd
    )
    with torch.no_grad():
        for a in adapters:
            a.up_q.normal_()
    sd = dec.trainable_state_dict()
    ref = dec(z, 1).detach()

    fresh = _decoder(num_levels=2)
    with pytest.raises(RuntimeError):
        fresh.load_trainable_state_dict(sd)
    fresh.enable_lora(rank=2)
    fresh.trunk.load_state_dict(dec.trunk.state_dict(), strict=False)  # same frozen weights
    fresh.load_trainable_state_dict(sd)
    assert torch.allclose(fresh(z, 1), ref, atol=1e-5)


def test_build_hierarchy_l1_indices_and_rollout() -> None:
    from point_hjepa.models.point_hjepa import PointHJEPA

    torch.manual_seed(0)
    wm = PointHJEPA(
        num_levels=3,
        strides=(2, 2),
        context_lengths=(3, 3, 3),
        backbone_pretrained=False,
        lambda_sigreg=(0.1, 0.1, 0.1),
        gamma_idm=(1.0, 1.0, 1.0),
        sigreg_num_slices=16,
        predictor_depth=1,
    ).eval()
    B, T = 2, wm.min_l1_frames
    z1 = torch.randn(B, T, 256)
    u1 = torch.randn(B, T - 1, 24)
    hier = wm.build_hierarchy(z1, u1)
    assert [i.tolist() for i in hier["l1_indices"]] == [list(range(13)), [0, 2, 4, 6, 8, 10, 12], [0, 4, 8, 12]]
    # level-2 state k is the encoder applied to the Level-1 state it indexes
    assert torch.allclose(hier["states"][1], wm.state_encoders[0](z1[:, hier["l1_indices"][1]]), atol=1e-6)

    with torch.no_grad():
        for lvl in range(3):
            z, a = hier["states"][lvl], hier["actions"][lvl]
            C, T_l = 3, z.shape[1]
            imag = wm.rollout(lvl, z[:, :C], a)
            assert imag.shape == (B, T_l - C, 256)
            one = wm.one_step_predictions(lvl, z, a)
            assert one.shape == (B, T_l - 1, 256)
            # the first imagined state is the one-step prediction from the observed context
            assert torch.allclose(imag[:, 0], one[:, C - 1], atol=1e-5)
            # within one training-length clip, sliding windows equal the causal-window forward at position C-1
            full = wm.predictors[lvl](z[:, :C], a[:, :C])
            assert torch.allclose(one[:, C - 1], full[:, C - 1], atol=1e-5)


_HAVE_PAGE4D = DEFAULT_PAGE4D_CKPT.is_file() and (DEFAULT_PAGE4D_ROOT / "model").is_dir()


@pytest.mark.skipif(
    not (_HAVE_PAGE4D and torch.cuda.is_available() and os.environ.get("PAGE4D_SMOKE") == "1"),
    reason="Real PAGE-4D trunk smoke test: needs scripts/setup_page4d.py, CUDA and PAGE4D_SMOKE=1",
)
def test_real_page4d_trunk_smoke() -> None:
    from point_hjepa.models.pc_decoder.page4d_trunk import PAGE4DTrunk

    device = torch.device("cuda")
    trunk = PAGE4DTrunk(grid=16)
    dec = HierarchicalPointDecoder(trunk, num_levels=2, latent_dim=256, target_center=(0.48, 0.0, 0.25)).to(device)
    assert sum(p.numel() for p in dec.trunk.parameters() if p.requires_grad) == 0
    z = torch.randn(2, 256, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        dense = dec(z, torch.tensor([0, 1], device=device))
    assert dense.shape == (2, 224 * 224, 6) and torch.isfinite(dense).all()
    dec.enable_lora(4)
    assert len(dec.adapters) == 72
    dec.to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = dec.loss(dec(z, 1), torch.rand(2, 512, 6, device=device), num_points=512)
    loss.backward()
    assert all(a.up_q.grad is not None for a in dec.adapters)
