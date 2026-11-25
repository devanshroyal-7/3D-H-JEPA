"""Spatial-to-latent Attentive Pooling Bridge with ragged padding masks and attention export."""

from __future__ import annotations

import torch
from torch import nn


def pack_ragged_tokens(
    tokens: torch.Tensor,
    offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert ragged ``[M_total, D]`` tokens + cumulative ``offset [N]`` to padded batches.

    Returns:
        padded: ``[N, M_max, D]``
        key_padding_mask: ``[N, M_max]`` boolean (``True`` indicates padded position)
        lengths: ``[N]`` token counts per frame.
    """
    device = tokens.device
    offset = offset.to(device=device, dtype=torch.long)
    starts = torch.cat([offset.new_zeros(1), offset[:-1]])
    lengths = offset - starts
    N = int(offset.shape[0])
    M_max = int(lengths.max().item()) if N > 0 else 0

    padded = tokens.new_zeros((N, M_max, tokens.shape[-1]))
    key_padding_mask = torch.ones((N, M_max), device=device, dtype=torch.bool)
    if tokens.shape[0] == 0:
        return padded, key_padding_mask, lengths

    frame_ids = torch.repeat_interleave(torch.arange(N, device=device), lengths)
    local_idx = torch.arange(tokens.shape[0], device=device) - torch.repeat_interleave(starts, lengths)
    padded[frame_ids, local_idx] = tokens
    key_padding_mask[frame_ids, local_idx] = False
    return padded, key_padding_mask, lengths


class _CrossAttentionBlock(nn.Module):
    """Pre-LN Multi-Head Cross-Attention + residual MLP block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_norm = self.norm_q(queries)
        kv_norm = self.norm_kv(kv)
        attn_out, attn_weights = self.attn(
            q_norm,
            kv_norm,
            kv_norm,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        x = queries + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x, attn_weights  # attn_weights: [N, K, M_max]


class AttentivePoolerBridge(nn.Module):
    """Compresses variable-length per-frame point tokens into fixed latent vectors ``z^(1)``."""

    def __init__(
        self,
        dim: int = 256,
        num_queries: int = 4,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_queries = int(num_queries)

        self.queries = nn.Parameter(torch.randn(1, self.num_queries, self.dim) * 0.02)
        self.blocks = nn.ModuleList(
            [_CrossAttentionBlock(self.dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        flat_dim = self.num_queries * self.dim  # 1024
        self.projector = nn.Sequential(
            nn.Linear(flat_dim, flat_dim // 2),
            nn.GELU(),
            nn.Linear(flat_dim // 2, self.dim),
            nn.LayerNorm(self.dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool ragged point tokens into ``z^(1) [N, D]`` and export attention weights ``[N, M_max]``."""
        padded, key_padding_mask, lengths = pack_ragged_tokens(tokens, offset)
        N = padded.shape[0]
        q = self.queries.expand(N, -1, -1)

        attn_weights = padded.new_zeros((N, self.num_queries, padded.shape[1]))
        for block in self.blocks:
            q, attn_weights = block(q, padded, key_padding_mask)

        # Average attention across K queries and mask padded tokens cleanly.
        scalar_attn = attn_weights.mean(dim=1)  # [N, M_max]
        scalar_attn = scalar_attn.masked_fill(key_padding_mask, 0.0)
        denom = scalar_attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        scalar_attn = scalar_attn / denom

        z1 = self.projector(q.reshape(N, -1))
        return z1, scalar_attn, lengths
