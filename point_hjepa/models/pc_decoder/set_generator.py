# Ported from JEPA4R (https://github.com/AIGeeksGroup/JEPA4R, jepa4r/decoder/set_generator.py), Apache-2.0.
"""FiLM set generator: latent ``z`` -> ``G*G`` tokens that replace PAGE-4D's image patch tokens.

256 learnable queries -> 3 residual MLP blocks modulated by ``z`` through FiLM -> 16 x 16 grid of 1,024-d tokens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FiLMBlock(nn.Module):
    """Residual per-element MLP modulated by the latent (FiLM)."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # h: (B, M, dim); gamma / beta: (B, 1, dim)
        return h + self.fc2(F.gelu(self.fc1(self.norm(h) * (1.0 + gamma) + beta)))


class SetDecoder(nn.Module):
    """latent ``(B, D)`` -> ``(B, M, out_dim)``.

    M learned queries say "which element of the set am I"; the latent modulates every block. Elements are decoded
    independently of each other. The last projection runs in float32.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_queries: int = 256,
        out_dim: int = 1024,
        dim: int = 512,
        hidden: int = 1024,
        depth: int = 3,
        query_dim: int = 128,
    ) -> None:
        super().__init__()
        self.dim, self.depth = dim, depth
        self.queries = nn.Parameter(torch.randn(num_queries, query_dim) * 0.02)
        self.q_proj = nn.Linear(query_dim, dim)
        self.in_norm = nn.LayerNorm(latent_dim)
        self.cond = nn.Sequential(nn.Linear(latent_dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU())
        self.film = nn.Linear(dim, 2 * dim * depth)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.blocks = nn.ModuleList([FiLMBlock(dim, hidden) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, out_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B = latent.size(0)
        mods = self.film(self.cond(self.in_norm(latent))).view(B, self.depth, 2, 1, self.dim)
        h = self.q_proj(self.queries).unsqueeze(0).expand(B, -1, -1)
        for i, blk in enumerate(self.blocks):
            h = blk(h, mods[:, i, 0], mods[:, i, 1])
        h = self.out_norm(h)
        with torch.autocast(device_type=h.device.type, enabled=False):
            return self.out(h.float())
