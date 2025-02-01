"""
Trading Environment Module
Provides a modular, efficient implementation of the trading environment.
"""

from .trading_env import TradingEnv
from typing import Dict, Any, Optional, Union, List
import pandas as pd

def create_trading_env(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    initial_balance: float = 10000,
    transaction_cost: float = 0.01,
    position_size: float = 0.2,
    use_position_profit: bool = False,
    use_holding_bonus: bool = False,
    use_trading_penalty: bool = False,
    training_mode: bool = True,
    log_frequency: int = 30,
    stock_names: Optional[List[str]] = None
) -> TradingEnv:
    """
    Factory function to create a new trading environment instance with given configuration.

    Args:
        data: Single or multiple asset price data
        initial_balance: Starting portfolio balance
        transaction_cost: Cost per trade as a fraction
        position_size: Maximum position size as a fraction of balance
        use_position_profit: Whether to include position profit in reward
        use_holding_bonus: Whether to reward holding profitable positions
        use_trading_penalty: Whether to penalize excessive trading
        training_mode: If True, enables training-specific features
        log_frequency: How often to log portfolio stats
        stock_names: List of stock symbols being traded

    Returns:
        Configured TradingEnv instance
    """
    return TradingEnv(
        data=data,
        initial_balance=initial_balance,
        transaction_cost=transaction_cost,
        position_size=position_size,
        use_position_profit=use_position_profit,
        use_holding_bonus=use_holding_bonus,
        use_trading_penalty=use_trading_penalty,
        training_mode=training_mode,
        log_frequency=log_frequency,
        stock_names=stock_names
    )

__all__ = ['TradingEnv', 'create_trading_env']