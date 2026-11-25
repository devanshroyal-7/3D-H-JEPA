"""Modular H-JEPA level components: encoders, action poolers, AdaLN-Zero predictor, IDM."""

from __future__ import annotations

import torch
from torch import nn


class ActionEncoderL1(nn.Module):
    """Maps grouped Level-1 primitive actions ``u_t^(1) [B, T, frame_skip * d_a]`` -> ``d_act1``."""

    def __init__(self, in_dim: int, out_dim: int = 64) -> None:
        super().__init__()
        hidden = max(out_dim * 2, in_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


class UpperStateEncoder(nn.Module):
    """2-layer MLP ``E^(l)`` mapping ``z^(l-1)`` (256) -> ``z^(l)`` (256)."""

    def __init__(self, dim: int = 256, hidden_dim: int = 384) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, z_prev: torch.Tensor) -> torch.Tensor:
        return self.net(z_prev)


class UpperActionPooler(nn.Module):
    """Condenses ``stride`` lower-level action embeddings into a macro-action ``a^(l)``."""

    def __init__(
        self,
        in_dim: int,
        stride: int,
        out_dim: int = 8,
        pooler_type: str = "transformer",
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.stride = int(stride)
        self.out_dim = int(out_dim)
        self.pooler_type = pooler_type

        if pooler_type == "transformer":
            proj_dim = max(64, in_dim)
            self.in_proj = nn.Linear(in_dim, proj_dim) if in_dim != proj_dim else nn.Identity()
            self.cls_token = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.02)
            self.pos_emb = nn.Parameter(torch.randn(1, stride + 1, proj_dim) * 0.02)
            self.layer = nn.TransformerEncoderLayer(
                d_model=proj_dim,
                nhead=num_heads,
                dim_feedforward=proj_dim * 2,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.out_proj = nn.Sequential(
                nn.LayerNorm(proj_dim),
                nn.Linear(proj_dim, out_dim),
            )
        elif pooler_type == "mlp":
            flat = in_dim * stride
            hidden = max(128, flat // 2)
            self.net = nn.Sequential(
                nn.Linear(flat, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_dim),
                nn.LayerNorm(out_dim),
            )
        else:
            raise ValueError(f"Unknown action pooler_type {pooler_type!r}")

    def forward(self, actions_chunk: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions_chunk: ``[B, K, stride, in_dim]``
        Returns:
            ``[B, K, out_dim]``
        """
        B, K, S, D = actions_chunk.shape
        if self.pooler_type == "mlp":
            return self.net(actions_chunk.reshape(B, K, S * D))

        x = self.in_proj(actions_chunk.reshape(B * K, S, D))
        cls = self.cls_token.expand(B * K, -1, -1)
        tokens = torch.cat([cls, x], dim=1) + self.pos_emb[:, : S + 1]
        out = self.layer(tokens)
        return self.out_proj(out[:, 0]).reshape(B, K, self.out_dim)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class AdaLNZeroBlock(nn.Module):
    """Causal self-attention + MLP block modulated via AdaLN-Zero from action ``a_t``."""

    def __init__(self, dim: int, num_heads: int, cond_dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(cond).chunk(
            6, dim=-1
        )
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + gate_msa * attn_out
        h_mlp = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(h_mlp)
        return x


class CausalAdaLNPredictor(nn.Module):
    """6-layer Causal AdaLN-Zero Transformer predicting next-step latent ``z_{t+1}``."""

    def __init__(
        self,
        dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        cond_dim: int = 64,
        mlp_ratio: float = 4.0,
        context_len: int = 3,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.context_len = int(context_len)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [AdaLNZeroBlock(dim, num_heads, cond_dim, mlp_ratio) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(dim)
        self.projector = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def _causal_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # i - j
        allowed = (diff >= 0) & (diff < self.context_len)
        return ~allowed  # True = masked out

    def forward(self, z_in: torch.Tensor, a_in: torch.Tensor) -> torch.Tensor:
        """Predict ``z_{t+1}`` for each ``t`` given ``z_{1:T-1}`` and ``a_{1:T-1}``."""
        T = z_in.shape[1]
        if T > self.pos_emb.shape[1]:
            raise ValueError(
                f"Sequence length {T} exceeds predictor max_seq_len={self.pos_emb.shape[1]}"
            )
        x = z_in + self.pos_emb[:, :T]
        mask = self._causal_window_mask(T, x.device)
        for block in self.blocks:
            x = block(x, a_in, mask)
        return self.projector(self.final_norm(x))


class InverseDynamicsHead(nn.Module):
    """2-layer MLP ``I^(l)`` predicting action ``a_t^(l)`` from ``[z_t^(l), z_{t+1}^(l)]``."""

    def __init__(self, state_dim: int = 256, hidden_dim: int = 384, out_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z_curr: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_curr, z_next], dim=-1))
