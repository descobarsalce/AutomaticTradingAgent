"""Factories for environment construction and checkpoint management."""
from __future__ import annotations

from typing import Any

from src.environment import TradingEnv
from src.core.agent_interfaces import EnvConfig


def build_trading_env(config: EnvConfig) -> TradingEnv:
    """Instantiate a TradingEnv using only validated manager outputs."""
    if config.provider is None:
        raise ValueError("EnvConfig.provider must be supplied")

    return TradingEnv(
        stock_names=config.stock_names,
        start_date=config.start_date,
        end_date=config.end_date,
        feature_config=config.feature_config,
        feature_pipeline=config.feature_pipeline,
        training_mode=config.training_mode,
        provider=config.provider,
        initial_balance=config.params.get("initial_balance", 10000),
        transaction_cost=config.params.get("transaction_cost", 0.0),
        max_pct_position_by_asset=config.params.get("max_pct_position_by_asset", 0.2),
        use_position_profit=config.params.get("use_position_profit", False),
        use_holding_bonus=config.params.get("use_holding_bonus", False),
        use_trading_penalty=config.params.get("use_trading_penalty", False),
        observation_days=config.params.get("history_length", 3),
        burn_in_days=config.params.get("burn_in_days", 20),
    )
