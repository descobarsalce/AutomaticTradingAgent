"""Token-based market model with contrastive and auxiliary objectives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.token_schema import TokenBatch, TokenSchema
from core.risk_budget_manager import RiskBudgetManager


def _pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: Sequence[float]) -> torch.Tensor:
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - pred[..., i]
        losses.append(torch.maximum((q - 1) * errors, q * errors))
    return torch.stack(losses, dim=-1).mean()


@dataclass
class ContrastiveLossConfig:
    temperature: float = 0.5
    weight: float = 1.0


class MarketTokenModel(nn.Module):
    """Transformer-style model operating on global and symbol tokens."""

    def __init__(
        self,
        schema: TokenSchema,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
        contrastive: ContrastiveLossConfig = ContrastiveLossConfig(),
    ) -> None:
        super().__init__()
        self.schema = schema
        self.quantiles = quantiles
        self.contrastive_cfg = contrastive
        self.global_encoder = nn.Linear(schema.global_feature_dim, d_model)
        self.symbol_encoder = nn.Linear(schema.symbol_feature_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.global_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.symbol_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.realized_vol_head = nn.Linear(d_model, 1)
        self.correlation_proxy_head = nn.Linear(d_model, schema.num_symbols)
        self.dispersion_head = nn.Linear(d_model, 1)
        self.liquidity_regime_head = nn.Linear(d_model, 3)
        self.quantile_head = nn.Linear(d_model, len(quantiles))
        self.idiosyncratic_vol_head = nn.Linear(d_model, 1)

        self.drift_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.register_buffer("running_global_mean", torch.zeros(d_model))
        self.risk_budget_manager: Optional[RiskBudgetManager] = None

    def attach_risk_budget_manager(self, manager: RiskBudgetManager) -> None:
        self.risk_budget_manager = manager

    def forward(self, batch: TokenBatch) -> Dict[str, torch.Tensor]:
        batch.validate_against(self.schema)
        bsz = batch.global_tokens.shape[0]

        global_embed = self.global_encoder(batch.global_tokens.view(bsz, -1, self.schema.global_feature_dim))
        z_global = global_embed.mean(dim=1)

        drift_signal = z_global - self.running_global_mean
        if batch.drift_features is not None:
            drift_signal = torch.cat([drift_signal, batch.drift_features], dim=-1)
        else:
            drift_signal = torch.cat([drift_signal, z_global], dim=-1)
        gate = self.drift_gate(drift_signal)
        z_global = z_global * gate

        symbol_embed = self.symbol_encoder(
            batch.symbol_tokens.view(bsz, self.schema.symbol_token_count, self.schema.symbol_feature_dim)
        )
        transformer_input = torch.cat([z_global.unsqueeze(1), symbol_embed], dim=1)
        encoded = self.transformer(transformer_input)
        z_global_post = encoded[:, 0]
        z_symbols = encoded[:, 1:].view(
            bsz, self.schema.num_symbols, self.schema.symbol_tokens_per_symbol, -1
        ).mean(dim=2)

        projections = {
            "z_global": z_global_post,
            "z_symbols": z_symbols,
            "global_proj": self.global_proj(z_global_post),
            "symbol_proj": self.symbol_proj(z_symbols),
        }

        outputs = {
            **projections,
            "realized_volatility": self.realized_vol_head(z_global_post).squeeze(-1),
            "dispersion": self.dispersion_head(z_global_post).squeeze(-1),
            "liquidity_regime": self.liquidity_regime_head(z_global_post),
            "correlation_proxy": self.correlation_proxy_head(z_global_post),
            "return_quantiles": self.quantile_head(z_symbols),
            "idiosyncratic_vol": self.idiosyncratic_vol_head(z_symbols).squeeze(-1),
            "symbol_volatility": self.idiosyncratic_vol_head(z_symbols).squeeze(-1),
            "drift_gate": gate,
        }

        if self.risk_budget_manager is not None:
            self.risk_budget_manager.ingest_forecasts(
                volatility=outputs["symbol_volatility"].detach().cpu().numpy(),
                correlation=outputs["correlation_proxy"].detach().cpu().numpy(),
                dispersion=outputs["dispersion"].detach().cpu().numpy(),
            )

        self.running_global_mean = 0.9 * self.running_global_mean + 0.1 * z_global.detach().mean(dim=0)
        return outputs

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        aux_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        aux_weights = aux_weights or {}
        losses: Dict[str, torch.Tensor] = {}

        g = F.normalize(outputs["global_proj"], dim=-1)
        s = F.normalize(outputs["symbol_proj"], dim=-1)
        positive = (g.unsqueeze(1) * s).sum(dim=-1)
        logits = positive / self.contrastive_cfg.temperature
        contrastive_loss = -F.log_softmax(logits, dim=-1).diag().mean()
        losses["contrastive"] = contrastive_loss * self.contrastive_cfg.weight

        losses["realized_vol"] = F.mse_loss(outputs["realized_volatility"], targets["realized_volatility"])
        losses["dispersion"] = F.mse_loss(outputs["dispersion"], targets["dispersion"])
        losses["correlation_proxy"] = F.mse_loss(outputs["correlation_proxy"], targets["correlation_proxy"])
        losses["liq_regime"] = F.cross_entropy(outputs["liquidity_regime"], targets["liquidity_regime"])

        q_loss = _pinball_loss(outputs["return_quantiles"], targets["return_quantiles"], self.quantiles)
        losses["quantiles"] = q_loss
        losses["idio_vol"] = F.mse_loss(outputs["idiosyncratic_vol"], targets["idiosyncratic_vol"])

        total = torch.zeros((), device=outputs["z_global"].device)
        for name, loss in losses.items():
            weight = aux_weights.get(name, 1.0)
            total = total + weight * loss
        return total, {k: v.item() for k, v in losses.items()}


__all__ = ["MarketTokenModel", "ContrastiveLossConfig"]
