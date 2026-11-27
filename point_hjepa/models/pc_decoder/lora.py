# LoRAQV / add_lora ported from JEPA4R (https://github.com/AIGeeksGroup/JEPA4R, jepa4r/decoder/lora.py), Apache-2.0.
# LoRAQVBank / add_lora_bank are the per-level extension used by the hierarchical decoder.
"""LoRA on fused ``qkv`` projections, update restricted to the q and v slices.

``LoRAQVBank`` holds one adapter per hierarchy level and picks it per sample from a shared ``LevelRouter``, so a
batch mixing latents of several levels runs through the shared frozen trunk in a single pass while every level keeps
its own adapters (identical to L separate LoRA-adapted decoders).
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRAQV(nn.Module):
    """``W x + s * [B_q; 0; B_v] (A x)``. The pretrained qkv weight is untouched; A is random, B is zero, so training
    starts exactly at the pretrained model."""

    def __init__(self, qkv: nn.Linear, rank: int, alpha: float | None = None) -> None:
        super().__init__()
        self.qkv, self.rank, self.scale = qkv, rank, (alpha or rank) / rank
        self.dim = qkv.in_features
        self.down = nn.Linear(self.dim, rank, bias=False)
        self.up_q = nn.Linear(rank, self.dim, bias=False)
        self.up_v = nn.Linear(rank, self.dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up_q.weight)
        nn.init.zeros_(self.up_v.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(x)
        delta = torch.cat(
            [self.up_q(h), torch.zeros_like(h[..., :1]).expand(*h.shape[:-1], self.dim), self.up_v(h)], dim=-1
        )
        return self.qkv(x) + self.scale * delta


def add_lora(module: nn.Module, rank: int) -> list[LoRAQV]:
    """Wrap every attention's fused qkv in the given blocks; returns the wrappers."""
    wrappers: list[LoRAQV] = []
    for name, child in list(module.named_children()):
        if name == "qkv" and isinstance(child, nn.Linear):
            w = LoRAQV(child, rank)
            setattr(module, name, w)
            wrappers.append(w)
        else:
            wrappers += add_lora(child, rank)
    return wrappers


class LevelRouter:
    """Shared, non-module holder of the per-sample level index ``(B,)`` the LoRA banks read during a trunk pass."""

    def __init__(self) -> None:
        self.idx: torch.Tensor | None = None


class LoRAQVBank(nn.Module):
    """``num_banks`` independent ``LoRAQV`` adapters on one frozen qkv; sample ``b`` uses bank ``router.idx[b]``.

    Each bank is initialised exactly like ``LoRAQV`` (kaiming-uniform ``down``, zero ``up_q`` / ``up_v``).
    """

    def __init__(self, qkv: nn.Linear, rank: int, num_banks: int, router: LevelRouter, alpha: float | None = None):
        super().__init__()
        self.qkv, self.rank, self.scale = qkv, rank, (alpha or rank) / rank
        self.dim = qkv.in_features
        self.num_banks = int(num_banks)
        self._router = router
        self.down = nn.Parameter(torch.empty(self.num_banks, rank, self.dim))
        self.up_q = nn.Parameter(torch.zeros(self.num_banks, self.dim, rank))
        self.up_v = nn.Parameter(torch.zeros(self.num_banks, self.dim, rank))
        for b in range(self.num_banks):
            nn.init.kaiming_uniform_(self.down.data[b], a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = self._router.idx
        if idx is None:
            raise RuntimeError("LoRAQVBank: router.idx must be set before the trunk forward")
        if idx.shape[0] != x.shape[0]:
            raise RuntimeError(f"LoRAQVBank: router.idx has {idx.shape[0]} entries for batch {x.shape[0]}")
        down, up_q, up_v = self.down[idx], self.up_q[idx], self.up_v[idx]  # (B, r, d), (B, d, r), (B, d, r)
        xs = x.reshape(x.shape[0], -1, x.shape[-1])
        h = torch.bmm(xs, down.transpose(1, 2).to(xs.dtype))  # (B, N, r)
        dq = torch.bmm(h, up_q.transpose(1, 2).to(h.dtype))
        dv = torch.bmm(h, up_v.transpose(1, 2).to(h.dtype))
        delta = torch.cat([dq, torch.zeros_like(dq), dv], dim=-1).reshape(*x.shape[:-1], 3 * self.dim)
        return self.qkv(x) + self.scale * delta


def add_lora_bank(module: nn.Module, rank: int, num_banks: int, router: LevelRouter) -> list[LoRAQVBank]:
    """Wrap every fused ``qkv`` linear under ``module`` with a per-level ``LoRAQVBank``; returns the wrappers."""
    wrappers: list[LoRAQVBank] = []
    for name, child in list(module.named_children()):
        if name == "qkv" and isinstance(child, nn.Linear):
            w = LoRAQVBank(child, rank, num_banks, router)
            setattr(module, name, w)
            wrappers.append(w)
        else:
            wrappers += add_lora_bank(child, rank, num_banks, router)
    return wrappers
