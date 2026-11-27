"""Per-level JEPA4R point-cloud decoders on one shared frozen PAGE-4D trunk.

Every hierarchy level ``l`` owns a full JEPA4R decoder head: a FiLM ``SetDecoder`` (latent -> 16 x 16 x 1024 tokens),
a learnable affine ``frame`` (PAGE-4D camera frame -> target frame, initialised as a virtual camera) and an rgb
branch, plus (stage B) its own rank-4 q/v LoRA bank on all 72 trunk attentions. The PAGE-4D weights are frozen, so
sharing one copy is exactly equivalent to L separate JEPA4R decoders.

Target frame: ``(xyz_world - target_center) * target_scale``, rgb in ``[0, 1]`` (JEPA4R's point-cloud affine).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from point_hjepa.models.pc_decoder.lora import LevelRouter, LoRAQVBank, add_lora_bank
from point_hjepa.models.pc_decoder.losses import chamfer, sliced_wasserstein
from point_hjepa.models.pc_decoder.page4d_trunk import inverse_log_transform, make_cam
from point_hjepa.models.pc_decoder.set_generator import SetDecoder

ADAPTER_SUFFIXES = (".down", ".up_q", ".up_v")


class LevelHead(nn.Module):
    """JEPA4R's new parameters for one level: set generator, affine ``frame`` and rgb branch."""

    def __init__(
        self,
        latent_dim: int,
        grid: int,
        token_dim: int,
        feat_dim: int,
        gen_dim: int = 512,
        gen_hidden: int = 1024,
        gen_depth: int = 3,
        query_dim: int = 128,
        color: bool = True,
        target_center: Sequence[float] = (0.0, 0.0, 0.0),
        target_scale: float = 4.0,
        view: Sequence[float] = (35.0, 30.0, 1.25),
    ) -> None:
        super().__init__()
        self.generator = SetDecoder(
            latent_dim=latent_dim,
            num_queries=grid * grid,
            out_dim=token_dim,
            dim=gen_dim,
            hidden=gen_hidden,
            depth=gen_depth,
            query_dim=query_dim,
        )
        self.rgb = (
            nn.Sequential(nn.Conv2d(feat_dim, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 3, 1))
            if color
            else None
        )
        # world -> camera, metres, OpenCV axes. JEPA4R looks at make_cam's default target; we look at target_center.
        cam = make_cam(az_deg=view[0], el_deg=view[1], dist=view[2], target=tuple(target_center), device="cpu")
        R, pos = cam["R"], -cam["R"].T @ cam["t"]
        # PAGE-4D camera frame (unit scene scale) -> target frame
        self.frame = nn.Linear(3, 3)
        with torch.no_grad():
            self.frame.weight.copy_(target_scale * view[2] * R.T)
            self.frame.bias.copy_(target_scale * (pos - torch.tensor(list(target_center), dtype=torch.float32)))

    def map_to_points(self, raw: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """``raw (B, 4, S, S)``, ``feats (B, C, S, S)`` -> dense ``(B, S*S, 3 or 6)`` in the target frame."""
        with torch.autocast(device_type=raw.device.type, enabled=False):
            xyz = self.frame(inverse_log_transform(raw[:, :3].float()).flatten(2).transpose(1, 2))
            if self.rgb is not None:
                xyz = torch.cat([xyz, torch.sigmoid(self.rgb(feats.float())).flatten(2).transpose(1, 2)], dim=-1)
        return xyz


class HierarchicalPointDecoder(nn.Module):
    """``num_levels`` JEPA4R decoders sharing one frozen trunk; ``(z, level) -> dense point map``."""

    def __init__(
        self,
        trunk: nn.Module,
        num_levels: int,
        latent_dim: int = 256,
        gen_dim: int = 512,
        gen_hidden: int = 1024,
        gen_depth: int = 3,
        query_dim: int = 128,
        color: bool = True,
        color_weight: float = 0.1,
        sw_weight: float = 1.0,
        sw_projections: int = 100,
        target_center: Sequence[float] = (0.0, 0.0, 0.0),
        target_scale: float = 4.0,
        view: Sequence[float] = (35.0, 30.0, 1.25),
    ) -> None:
        super().__init__()
        self.trunk = trunk
        self.num_levels = int(num_levels)
        self.color = bool(color)
        self.color_weight, self.sw_weight, self.sw_projections = float(color_weight), float(sw_weight), int(sw_projections)
        self.register_buffer("target_center", torch.tensor(list(target_center), dtype=torch.float32), persistent=True)
        self.register_buffer("target_scale", torch.tensor(float(target_scale)), persistent=True)
        self.heads = nn.ModuleList(
            [
                LevelHead(
                    latent_dim=latent_dim,
                    grid=trunk.grid,
                    token_dim=trunk.embed_dim,
                    feat_dim=trunk.feat_dim,
                    gen_dim=gen_dim,
                    gen_hidden=gen_hidden,
                    gen_depth=gen_depth,
                    query_dim=query_dim,
                    color=color,
                    target_center=target_center,
                    target_scale=target_scale,
                    view=view,
                )
                for _ in range(self.num_levels)
            ]
        )
        self.router = LevelRouter()
        self.adapters: list[LoRAQVBank] = []

    # ---------------------------------------------------------------- stage B
    def enable_lora(self, rank: int = 4) -> list[LoRAQVBank]:
        """Add a per-level rank-``rank`` q/v LoRA bank on every fused qkv of the trunk's block lists."""
        if self.adapters:
            return self.adapters
        for blocks in self.trunk.lora_block_lists():
            self.adapters += add_lora_bank(blocks, rank, self.num_levels, self.router)
        return self.adapters

    def lora_parameters(self) -> list[nn.Parameter]:
        return [p for a in self.adapters for p in (a.down, a.up_q, a.up_v)]

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.heads.parameters())

    # ---------------------------------------------------------- checkpoints
    @staticmethod
    def _is_frozen_key(key: str) -> bool:
        return key.startswith("trunk.") and not key.endswith(ADAPTER_SUFFIXES)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Everything except the frozen PAGE-4D weights (those come back from the released checkpoint)."""
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if not self._is_frozen_key(k)}

    def load_trainable_state_dict(self, sd: dict[str, torch.Tensor], strict: bool = True) -> None:
        if any(k.endswith(ADAPTER_SUFFIXES) for k in sd) and not self.adapters:
            raise RuntimeError("Checkpoint holds LoRA adapters: call enable_lora(rank) before loading it")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not self._is_frozen_key(k)]
        if strict and (missing or unexpected):
            raise RuntimeError(f"Decoder state mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")

    def train(self, mode: bool = True) -> "HierarchicalPointDecoder":
        super().train(mode)
        self.trunk.train(mode)  # the trunk pins its frozen parts to eval
        return self

    # ------------------------------------------------------------- forward
    def _level_tensor(self, level: int | torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
        if isinstance(level, int):
            if not 0 <= level < self.num_levels:
                raise ValueError(f"level {level} out of range for {self.num_levels} levels")
            return torch.full((batch,), level, dtype=torch.long, device=device)
        level = level.to(device=device, dtype=torch.long)
        if level.shape != (batch,):
            raise ValueError(f"level must be an int or a (B,) tensor, got shape {tuple(level.shape)}")
        return level

    def forward(self, z: torch.Tensor, level: int | torch.Tensor) -> torch.Tensor:
        """``z (B, D)`` decoded by the decoder of ``level`` (int or per-sample ``(B,)``) ->
        dense ``(B, side*side, 6)`` point map in the target frame (xyz, rgb in [0, 1])."""
        B = z.shape[0]
        lvl = self._level_tensor(level, B, z.device)
        present = [int(l) for l in torch.unique(lvl).tolist()]

        tokens: torch.Tensor | None = None
        masks = {l: lvl == l for l in present}
        for l in present:
            t_l = self.heads[l].generator(z[masks[l]])
            if tokens is None:
                tokens = t_l.new_zeros((B, *t_l.shape[1:]))
            tokens = tokens.index_put((masks[l].nonzero(as_tuple=True)[0],), t_l)

        # Left set after the pass: activation checkpointing re-runs the trunk blocks (and so the LoRA banks) during
        # backward, which must see the same per-sample levels. Hence: one decoder forward per backward.
        self.router.idx = lvl
        raw, feats = self.trunk.tokens_to_map(tokens)

        dense: torch.Tensor | None = None
        for l in present:
            p_l = self.heads[l].map_to_points(raw[masks[l]], feats[masks[l]])
            if dense is None:
                dense = p_l.new_zeros((B, *p_l.shape[1:]))
            dense = dense.index_put((masks[l].nonzero(as_tuple=True)[0],), p_l)
        return dense

    # ------------------------------------------------------------- helpers
    @staticmethod
    def pick(dense: torch.Tensor, n: int, fixed: bool) -> torch.Tensor:
        """``n`` pixels of the map: a fresh random subset while training (every pixel gets supervised), a fixed one
        (``torch.randperm`` with a seed-0 generator) otherwise. ``n`` > map size samples with replacement."""
        M = dense.size(1)
        g = torch.Generator(device=dense.device).manual_seed(0) if fixed else None
        if n <= M:
            idx = torch.randperm(M, device=dense.device, generator=g)[:n]
        else:
            idx = torch.randint(0, M, (n,), device=dense.device, generator=g)
        return dense[:, idx]

    def to_target(self, xyz: torch.Tensor, rgb: torch.Tensor | None = None) -> torch.Tensor:
        """World-frame metres (+ uint8-range rgb) -> target frame ``(..., 3 or 6)``."""
        out = (xyz.float() - self.target_center.to(xyz.device)) * self.target_scale.to(xyz.device)
        if rgb is not None and self.color:
            out = torch.cat([out, rgb.float() / 255.0], dim=-1)
        return out

    def to_world(self, pts: torch.Tensor) -> torch.Tensor:
        """Target frame -> world-frame metres (xyz) and rgb in [0, 255] (if present)."""
        xyz = pts[..., :3] / self.target_scale.to(pts.device) + self.target_center.to(pts.device)
        if pts.size(-1) > 3:
            return torch.cat([xyz, pts[..., 3:].clamp(0, 1) * 255.0], dim=-1)
        return xyz

    def decode_points(self, z: torch.Tensor, level: int | torch.Tensor, num_points: int) -> torch.Tensor:
        """Fixed-subset ``(B, num_points, 6)`` decoded cloud in world-frame metres (rgb 0-255)."""
        return self.to_world(self.pick(self(z, level), num_points, fixed=True))

    def loss(
        self,
        dense: torch.Tensor,
        target: torch.Tensor,
        num_points: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """JEPA4R Eq. 2 between ``num_points`` picked pixels of ``dense`` and ``target (B, M, 3 or 6)``."""
        pred = self.pick(dense, num_points, fixed=not self.training)
        loss, logs = chamfer(pred, target, self.color_weight)
        if self.sw_weight > 0:
            sw = sliced_wasserstein(pred, target, self.sw_projections)
            loss = loss + self.sw_weight * sw
            logs["sw"] = sw.detach()
        return loss, logs
