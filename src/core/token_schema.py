"""Token schema utilities for market tokenization.

Defines a simple schema to represent global tokens and per-symbol tokens
used by the MarketTokenModel. The schema validates dimensions and helps
construct batches for model consumption.
"""
from __future__ import annotations

from dataclasses import src.dataclass
from typing import Optional, Tuple
import torch


@dataclass
class TokenSchema:
    """Specification for tokenized market inputs.

    Attributes:
        global_feature_dim: Dimension of the global token feature vector.
        symbol_feature_dim: Dimension of the per-symbol token features.
        num_symbols: Number of tradable symbols in the batch.
        global_tokens: Number of global tokens (defaults to 1 for a market-wide summary).
        symbol_tokens_per_symbol: Number of tokens allocated to each symbol (defaults to 1).
    """

    global_feature_dim: int
    symbol_feature_dim: int
    num_symbols: int
    global_tokens: int = 1
    symbol_tokens_per_symbol: int = 1

    def __post_init__(self) -> None:
        if self.global_feature_dim <= 0:
            raise ValueError("global_feature_dim must be positive")
        if self.symbol_feature_dim <= 0:
            raise ValueError("symbol_feature_dim must be positive")
        if self.num_symbols <= 0:
            raise ValueError("num_symbols must be positive")
        if self.global_tokens <= 0:
            raise ValueError("global_tokens must be positive")
        if self.symbol_tokens_per_symbol <= 0:
            raise ValueError("symbol_tokens_per_symbol must be positive")

    @property
    def symbol_token_count(self) -> int:
        """Total number of symbol tokens in a batch."""
        return self.num_symbols * self.symbol_tokens_per_symbol

    @property
    def global_shape(self) -> Tuple[int, ...]:
        return (self.global_tokens, self.global_feature_dim)

    @property
    def symbol_shape(self) -> Tuple[int, ...]:
        return (self.num_symbols, self.symbol_tokens_per_symbol, self.symbol_feature_dim)


@dataclass
class TokenBatch:
    """Container for model-ready tokenized inputs."""

    global_tokens: torch.Tensor
    symbol_tokens: torch.Tensor
    drift_features: Optional[torch.Tensor] = None

    def validate_against(self, schema: TokenSchema) -> None:
        """Validate shapes against a schema before sending to the model."""
        if self.global_tokens.shape[1:] != schema.global_shape:
            raise ValueError(
                f"global token shape {self.global_tokens.shape[1:]} does not match schema {schema.global_shape}"
            )
        if self.symbol_tokens.shape[1:] != schema.symbol_shape:
            raise ValueError(
                f"symbol token shape {self.symbol_tokens.shape[1:]} does not match schema {schema.symbol_shape}"
            )
        if self.drift_features is not None and self.drift_features.shape[0] != self.global_tokens.shape[0]:
            raise ValueError("drift_features batch dimension must match global_tokens")


__all__ = ["TokenSchema", "TokenBatch"]
