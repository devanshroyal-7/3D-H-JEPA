# ``tokens_to_map`` / ``load_page4d`` / ``make_cam`` ported from JEPA4R (https://github.com/AIGeeksGroup/JEPA4R,
# jepa4r/decoder/backbones/page4d.py and jepa4r/decoder/head.py), Apache-2.0.
"""Frozen PAGE-4D trunk shared by every level's decoder.

PAGE-4D [Zhou et al., ICLR 2026] = DINOv2 ViT-L + 24 alternating frame / global attention layers + DPT point head.
The set-generator tokens replace the output of DINOv2's conv patch projection (the only dropped part). The
aggregator runs with S = 1 and the DPT point head returns a 224 x 224 xyz + confidence map; the per-level heads then
apply their own affine ``frame`` and rgb branch to that map.

PAGE-4D's code: ``git clone https://github.com/kaichen-z/PAGE4D`` at ``c4a39ac`` (``scripts/setup_page4d.py``),
located via ``$PAGE4D_ROOT`` (default ``<repo>/third_party/PAGE4D``). Weights: ``checkpoint_nomask.pt`` from the HF
dataset ``zhouk777/PAGE4D``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAGE4D_ROOT = REPO_ROOT / "third_party" / "PAGE4D"
DEFAULT_PAGE4D_CKPT = REPO_ROOT / "checkpoints" / "page4d" / "checkpoint_nomask.pt"


def inverse_log_transform(y: torch.Tensor) -> torch.Tensor:
    """``sign(y) * (exp(|y|) - 1)``: PAGE-4D / VGGT point-head activation (``page.heads.head_act``)."""
    return torch.sign(y) * torch.expm1(torch.abs(y))


def make_cam(
    az_deg: float = 45.0,
    el_deg: float = 40.0,
    dist: float = 1.2,
    target: tuple[float, float, float] | list[float] = (0.36, 0.0, 0.2),
    fovy_deg: float = 45.0,
    size: int = 320,
    device: str | torch.device = "cpu",
) -> dict:
    """World -> camera rotation / translation and intrinsics (OpenCV axes: x right, y down, z forward)."""
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    pos = np.asarray(target, float) + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    fwd = np.asarray(target, float) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(fwd, right), fwd])
    f = 0.5 * size / np.tan(np.deg2rad(fovy_deg) / 2)
    return dict(
        R=torch.tensor(R, dtype=torch.float32, device=device),
        t=torch.tensor(-R @ pos, dtype=torch.float32, device=device),
        f=f,
        c=size / 2,
        size=size,
        device=device,
    )


def _import_page4d(page4d_root: str | Path | None):
    root = Path(page4d_root or os.environ.get("PAGE4D_ROOT") or DEFAULT_PAGE4D_ROOT)
    model_dir = root / "model"
    if not (model_dir / "page" / "models" / "vggt.py").is_file():
        raise FileNotFoundError(
            f"PAGE-4D code not found under {root} (expected model/page/models/vggt.py); run scripts/setup_page4d.py"
        )
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from page.models.vggt import VGGT as PAGE4D  # noqa: PLC0415

    return PAGE4D


def load_page4d(weights: str | Path | None = None, page4d_root: str | Path | None = None) -> nn.Module:
    """PAGE-4D with its point head only, released weights loaded (no missing keys allowed)."""
    PAGE4D = _import_page4d(page4d_root)
    m = PAGE4D(enable_camera=False, enable_depth=False, enable_track=False)
    weights = Path(weights) if weights is not None else DEFAULT_PAGE4D_CKPT
    if not weights.is_file():
        raise FileNotFoundError(f"PAGE-4D weights not found: {weights}; run scripts/setup_page4d.py")
    sd = torch.load(weights, map_location="cpu", weights_only=False, mmap=True)
    sd = sd.get("model", sd)
    missing, _unexpected = m.load_state_dict(sd, strict=False)
    assert not missing, f"PAGE-4D weights missing: {missing[:5]}"
    return m


class PAGE4DTrunk(nn.Module):
    """Frozen DINOv2 + aggregator + DPT point head; ``tokens (B, G*G, 1024) -> raw map, head features``.

    Attributes used by the per-level heads: ``embed_dim`` (token width), ``grid``, ``patch_size``, ``side``
    (``grid * patch_size`` = 224) and ``feat_dim`` (channels entering ``output_conv2``, for the rgb branch).
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        page4d_root: str | Path | None = None,
        grid: int = 16,
        grad_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        m = load_page4d(weights, page4d_root)
        self.dino = m.aggregator.patch_embed
        self.dino.patch_embed = nn.Identity()  # the pixel -> token conv: the only dropped part
        m.aggregator.patch_embed = None
        # inert with the released nomask schedule; never called on the spliced path
        m.aggregator.spatial_mask_head = None
        self.aggregator, self.head = m.aggregator, m.point_head
        self.grid = int(grid)
        self.patch_size = int(self.aggregator.patch_size)
        self.side = self.grid * self.patch_size
        self.embed_dim = int(self.dino.embed_dim)
        self.feat_dim = int(self.head.scratch.output_conv2[0].in_channels)
        self.grad_checkpointing = bool(grad_checkpointing)
        self.requires_grad_(False)

    def lora_block_lists(self) -> list[nn.Module]:
        """The block lists whose fused qkv get LoRA in stage B: 24 DINOv2 + 24 frame + 24 global = 72 attentions."""
        return [self.dino.blocks, self.aggregator.frame_blocks, self.aggregator.global_blocks]

    def train(self, mode: bool = True) -> "PAGE4DTrunk":
        # frozen pretrained parts always stay in eval (adapters have no dropout / norm state)
        super().train(mode)
        for mod in (self.dino, self.aggregator, self.head):
            mod.eval()
        return self

    def _run(self, block: nn.Module, *args: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpointing and torch.is_grad_enabled():
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    def tokens_to_map(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens ``(B, G*G, C)`` in place of the patch-projection output ->
        ``raw (B, 4, side, side)`` (xyz in PAGE-4D's log space + confidence logits) and
        ``feats (B, feat_dim, side, side)`` (input of ``output_conv2``), both float32."""
        B, G, side = tokens.size(0), self.grid, self.side
        # DINOv2: prepare_tokens_with_masks without the conv
        d = self.dino
        x = torch.cat((d.cls_token.expand(B, -1, -1).to(tokens.dtype), tokens), dim=1)
        x = x + d.interpolate_pos_encoding(x, side, side)
        if d.register_tokens is not None:
            x = torch.cat((x[:, :1], d.register_tokens.expand(B, -1, -1).to(x.dtype), x[:, 1:]), dim=1)
        for blk in d.blocks:
            x = self._run(blk, x)
        patch_tokens = d.norm(x)[:, d.num_register_tokens + 1 :]

        # Aggregator.forward with S = 1, after its patch embed
        a = self.aggregator
        special = torch.cat([a.camera_token[:, 0], a.register_token[:, 0]], dim=1).expand(B, -1, -1)
        x = torch.cat([special.to(patch_tokens.dtype), patch_tokens], dim=1)
        pos = a.position_getter(B, G, G, device=x.device) + 1
        pos = torch.cat([pos.new_zeros(B, a.patch_start_idx, 2), pos], dim=1)
        # PAGE-4D's aggregator keeps every layer
        cached = getattr(a, "cached_layer_indices", None) or set(range(a.depth))
        taps: list[torch.Tensor | None] = []
        for i in range(a.depth):
            f = x = self._run(a.frame_blocks[i], x, pos)
            x = self._run(a.global_blocks[i], x, pos)
            taps.append(torch.cat([f, x], dim=-1)[:, None] if i in cached else None)

        # DPTHead._forward_impl up to the fused feature map
        h = self.head
        out = []
        for j, layer in enumerate(h.intermediate_layer_idx):
            t = h.norm(taps[layer][:, 0, a.patch_start_idx :])
            t = t.permute(0, 2, 1).reshape(B, t.size(-1), G, G)
            t = h.projects[j](t)
            if h.pos_embed:
                t = h._apply_pos_embed(t, side, side)
            out.append(h.resize_layers[j](t))
        feats = h.scratch_forward(out)
        feats = nn.functional.interpolate(feats, size=(side, side), mode="bilinear", align_corners=True)
        if h.pos_embed:
            feats = h._apply_pos_embed(feats, side, side)
        with torch.autocast(device_type=feats.device.type, enabled=False):  # coordinates in float32
            feats = feats.float()
            raw = h.scratch.output_conv2(feats)  # (B, 4, side, side): xyz + confidence
        return raw, feats
