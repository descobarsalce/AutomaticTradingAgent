"""Lightweight gating head for portfolio weight blending.

The gate takes a global latent representation (e.g., ``z_global``)
concatenated with risk features and produces a scalar ``g[t]`` in
``[0, 1]`` that can be used to blend previous holdings with proposed
portfolio weights.
"""

from typing import Optional

import torch
import torch.nn as nn


class GateHead(nn.Module):
    """Predict a gating scalar from global context.

    Args:
        global_dim: Dimensionality of the global latent vector.
        risk_feature_dim: Number of additional risk features.
        hidden_dim: Hidden size for the intermediate projection.
    """

    def __init__(
        self,
        global_dim: int,
        risk_feature_dim: int = 0,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(32, global_dim // 2)
        input_dim = global_dim + risk_feature_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, z_global: torch.Tensor, risk_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute ``g[t]`` from the provided context.

        Args:
            z_global: Tensor of shape ``(batch, global_dim)``.
            risk_features: Optional tensor of shape ``(batch, risk_feature_dim)``.

        Returns:
            Tensor of shape ``(batch, 1)`` with values clipped to ``[0, 1]``.
        """

        if risk_features is not None:
            z_global = torch.cat([z_global, risk_features], dim=-1)
        return self.net(z_global)
