"""Full modular L-level bottom-up Point-H-JEPA model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from point_hjepa.models.attentive_pooler import AttentivePoolerBridge
from point_hjepa.models.hjepa_level import (
    ActionEncoderL1,
    CausalAdaLNPredictor,
    InverseDynamicsHead,
    UpperActionPooler,
    UpperStateEncoder,
)
from point_hjepa.models.sigreg import SIGReg
from point_hjepa.models.utonia_backbone import UtoniaBackbone


class PointHJEPA(nn.Module):
    """Bottom-up unsegmented 6D point-cloud Hierarchical JEPA world model."""

    def __init__(
        self,
        num_levels: int = 2,
        strides: Sequence[int] = (2,),
        context_lengths: Sequence[int] = (3, 3),
        action_dim: int = 8,
        frame_skip: int = 3,
        latent_dim: int = 256,
        action_l1_dim: int = 64,
        macro_action_dim: int = 8,
        action_pooler_type: str = "transformer",
        backbone_preset: str = "utonia_tiny",
        backbone_pretrained: bool = True,
        serialization_depth: int = 12,
        freeze_stages: int = 3,
        fourier_bands: int = 8,
        utonia_scale: float = 4.0,
        pooler_queries: int = 4,
        pooler_heads: int = 4,
        pooler_layers: int = 2,
        pooler_mlp_ratio: float = 4.0,
        predictor_depth: int = 6,
        predictor_heads: int = 8,
        predictor_mlp_ratio: float = 4.0,
        lambda_sigreg: Sequence[float] = (0.08, 0.16),
        gamma_idm: Sequence[float] = (50.0, 50.0),
        sigreg_num_slices: int = 1024,
        sigreg_t_min: float = 0.2,
        sigreg_t_max: float = 4.0,
        sigreg_num_knots: int = 32,
    ) -> None:
        super().__init__()
        self.num_levels = int(num_levels)
        self.strides = [int(s) for s in strides]
        self.context_lengths = [int(c) for c in context_lengths]
        self.action_dim = int(action_dim)
        self.frame_skip = int(frame_skip)
        self.latent_dim = int(latent_dim)
        self.utonia_scale = float(utonia_scale)
        self.lambda_sigreg = [float(x) for x in lambda_sigreg]
        self.gamma_idm = [float(x) for x in gamma_idm]

        if len(self.strides) != self.num_levels - 1:
            raise ValueError(f"Expected {self.num_levels - 1} strides, got {len(self.strides)}")
        if len(self.context_lengths) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} context_lengths, got {len(self.context_lengths)}")
        if len(self.lambda_sigreg) != self.num_levels or len(self.gamma_idm) != self.num_levels:
            raise ValueError("lambda_sigreg and gamma_idm must match num_levels")

        # Level-1 perception & bridge
        self.backbone = UtoniaBackbone(
            preset=backbone_preset,
            pretrained=backbone_pretrained,
            serialization_depth=serialization_depth,
            freeze_stages=freeze_stages,
            out_dim=self.latent_dim,
            fourier_bands=fourier_bands,
        )
        self.pooler = AttentivePoolerBridge(
            dim=self.latent_dim,
            num_queries=pooler_queries,
            num_heads=pooler_heads,
            num_layers=pooler_layers,
            mlp_ratio=pooler_mlp_ratio,
        )

        # Level-1 action encoder, predictor, IDM
        u1_dim = self.frame_skip * self.action_dim
        self.action_l1 = ActionEncoderL1(in_dim=u1_dim, out_dim=action_l1_dim)
        self.predictors = nn.ModuleList(
            [
                CausalAdaLNPredictor(
                    dim=self.latent_dim,
                    depth=predictor_depth,
                    num_heads=predictor_heads,
                    cond_dim=action_l1_dim,
                    mlp_ratio=predictor_mlp_ratio,
                    context_len=self.context_lengths[0],
                )
            ]
        )
        self.idm_heads = nn.ModuleList(
            [InverseDynamicsHead(state_dim=self.latent_dim, hidden_dim=384, out_dim=u1_dim)]
        )

        # Upper levels (l >= 2)
        self.state_encoders = nn.ModuleList()
        self.action_poolers = nn.ModuleList()
        prev_act_dim = action_l1_dim
        for lvl_idx in range(1, self.num_levels):
            stride = self.strides[lvl_idx - 1]
            self.state_encoders.append(UpperStateEncoder(dim=self.latent_dim, hidden_dim=384))
            self.action_poolers.append(
                UpperActionPooler(
                    in_dim=prev_act_dim,
                    stride=stride,
                    out_dim=macro_action_dim,
                    pooler_type=action_pooler_type,
                )
            )
            self.predictors.append(
                CausalAdaLNPredictor(
                    dim=self.latent_dim,
                    depth=predictor_depth,
                    num_heads=predictor_heads,
                    cond_dim=macro_action_dim,
                    mlp_ratio=predictor_mlp_ratio,
                    context_len=self.context_lengths[lvl_idx],
                )
            )
            self.idm_heads.append(
                InverseDynamicsHead(
                    state_dim=self.latent_dim,
                    hidden_dim=384,
                    out_dim=macro_action_dim,
                )
            )
            prev_act_dim = macro_action_dim

        self.sigreg = SIGReg(
            num_slices=sigreg_num_slices,
            t_min=sigreg_t_min,
            t_max=sigreg_t_max,
            num_knots=sigreg_num_knots,
        )

    def get_param_groups(
        self, lr_backbone: float, lr_head: float, weight_decay: float
    ) -> list[dict[str, Any]]:
        backbone_params = [p for p in self.backbone.unfrozen_backbone_parameters() if p.requires_grad]
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [
            p for p in self.parameters() if p.requires_grad and id(p) not in backbone_ids
        ]
        groups: list[dict[str, Any]] = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
        if head_params:
            groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})
        return groups

    def encode_level1(
        self,
        point_data: Mapping[str, Any],
        batch_size: int | None = None,
        seq_len: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode collated point frames into Level-1 latents + attention metadata."""
        tokens, coord, offset = self.backbone(point_data)
        z_flat, attn_weights, lengths = self.pooler(tokens, offset)
        if batch_size is not None and seq_len is not None:
            z1 = z_flat.view(batch_size, seq_len, self.latent_dim)
        else:
            z1 = z_flat
        return {
            "z1": z1,
            "coord": coord,
            "offset": offset,
            "attn_weights": attn_weights,
            "lengths": lengths,
        }

    def forward(
        self,
        point_data: Mapping[str, Any],
        actions_u1: torch.Tensor,
        batch_size: int,
        seq_len: int,
        return_attn: bool = False,
    ) -> dict[str, Any]:
        """Forward pass across all L levels computing prediction, SIGReg, and IDM losses."""
        enc1 = self.encode_level1(point_data, batch_size=batch_size, seq_len=seq_len)
        z_curr = enc1["z1"]  # [B, T_1, D]
        a_curr = self.action_l1(actions_u1)  # [B, T_1 - 1, d_act1]

        states: list[torch.Tensor] = [z_curr]
        actions: list[torch.Tensor] = [a_curr]
        raw_actions: list[torch.Tensor] = [actions_u1]

        for lvl_idx in range(1, self.num_levels):
            stride = self.strides[lvl_idx - 1]
            T_prev = z_curr.shape[1]
            k_trans = (T_prev - 1) // stride
            if k_trans < 1:
                raise ValueError(
                    f"Level {lvl_idx + 1}: T_prev={T_prev} too short for stride={stride}"
                )
            state_indices = torch.arange(
                0, (k_trans + 1) * stride, stride, device=z_curr.device, dtype=torch.long
            )
            z_next = self.state_encoders[lvl_idx - 1](z_curr[:, state_indices])

            act_trim = a_curr[:, : k_trans * stride].reshape(
                batch_size, k_trans, stride, a_curr.shape[-1]
            )
            a_next = self.action_poolers[lvl_idx - 1](act_trim)

            states.append(z_next)
            actions.append(a_next)
            raw_actions.append(a_next)
            z_curr = z_next
            a_curr = a_next

        total_loss = z_curr.new_zeros((), dtype=torch.float32)
        metrics: dict[str, torch.Tensor] = {}

        for lvl in range(self.num_levels):
            z_l = states[lvl]
            a_l = actions[lvl]

            pred_z = self.predictors[lvl](z_l[:, :-1], a_l)
            loss_pred = F.mse_loss(pred_z.float(), z_l[:, 1:].float())
            loss_sig_z = self.sigreg(z_l)

            pred_a = self.idm_heads[lvl](z_l[:, :-1], z_l[:, 1:])
            if lvl == 0:
                target_a = raw_actions[0].float()
                loss_sig_a = z_l.new_zeros((), dtype=torch.float32)
            else:
                # Mandatory stop-gradient on learned macro-action target a^(l)
                target_a = a_l.detach().float()
                loss_sig_a = self.sigreg(a_l)

            loss_idm = F.mse_loss(pred_a.float(), target_a)

            lam = self.lambda_sigreg[lvl]
            gam = self.gamma_idm[lvl]
            lvl_loss = loss_pred + lam * loss_sig_z + (lam * loss_sig_a if lvl > 0 else 0.0) + gam * loss_idm
            total_loss = total_loss + lvl_loss

            lnum = lvl + 1
            metrics[f"l{lnum}/loss_pred"] = loss_pred
            metrics[f"l{lnum}/loss_sigreg_z"] = loss_sig_z
            if lvl > 0:
                metrics[f"l{lnum}/loss_sigreg_a"] = loss_sig_a
            metrics[f"l{lnum}/loss_idm"] = loss_idm
            metrics[f"l{lnum}/loss_total"] = lvl_loss

        metrics["loss"] = total_loss
        out: dict[str, Any] = {
            "loss": total_loss,
            "metrics": metrics,
            "states": states,
            "actions": actions,
        }
        if return_attn:
            out["coord"] = enc1["coord"]
            out["offset"] = enc1["offset"]
            out["attn_weights"] = enc1["attn_weights"]
            out["lengths"] = enc1["lengths"]
        return out
