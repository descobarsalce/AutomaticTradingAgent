"""Risk budgeting utilities that consume model risk forecasts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class RiskBudgetConfig:
    total_risk_budget: float = 1.0
    min_budget: float = 0.01
    max_budget: float = 0.3
    correlation_penalty: float = 0.25
    dispersion_floor: float = 0.05


@dataclass
class RiskBudgetManager:
    """Translate model risk forecasts into per-symbol risk budgets."""

    config: RiskBudgetConfig = field(default_factory=RiskBudgetConfig)
    latest_forecasts: Dict[str, torch.Tensor] = field(default_factory=dict)

    def ingest_forecasts(
        self,
        volatility: torch.Tensor,
        correlation: torch.Tensor,
        dispersion: Optional[torch.Tensor] = None,
    ) -> None:
        self.latest_forecasts["volatility"] = torch.as_tensor(volatility, dtype=torch.float32)
        self.latest_forecasts["correlation"] = torch.as_tensor(correlation, dtype=torch.float32)
        if dispersion is not None:
            self.latest_forecasts["dispersion"] = torch.as_tensor(dispersion, dtype=torch.float32)

    def compute_budgets(self) -> Dict[int, float]:
        if "volatility" not in self.latest_forecasts or "correlation" not in self.latest_forecasts:
            raise ValueError("RiskBudgetManager requires volatility and correlation forecasts before allocation")

        vol = self.latest_forecasts["volatility"]
        corr = self.latest_forecasts["correlation"]
        dispersion = self.latest_forecasts.get("dispersion")

        vol = torch.clamp(vol, min=1e-6)
        base_scores = 1.0 / vol
        if dispersion is not None:
            dispersion_signal = torch.maximum(dispersion.mean(), torch.tensor(self.config.dispersion_floor))
            base_scores = base_scores * dispersion_signal

        corr_penalty = 1 + self.config.correlation_penalty * torch.abs(corr)
        risk_scores = base_scores / corr_penalty

        raw = risk_scores.mean(dim=0) if risk_scores.ndim > 1 else risk_scores
        normalized = raw / raw.sum() * self.config.total_risk_budget
        clipped = torch.clamp(normalized, min=self.config.min_budget, max=self.config.max_budget)
        clipped = clipped / clipped.sum() * self.config.total_risk_budget

        return {i: float(budget) for i, budget in enumerate(clipped.cpu())}


__all__ = ["RiskBudgetManager", "RiskBudgetConfig"]
