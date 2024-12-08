from typing import Dict, Any, Optional, Union
from gymnasium import Env
from .base_agent import BaseAgent
from utils.common import (
    validate_numeric,
    validate_trading_params,
    MAX_POSITION_SIZE,
    MIN_POSITION_SIZE,
    DEFAULT_STOP_LOSS,
    DEFAULT_TAKE_PROFIT
)

class TradingAgent(BaseAgent):
    """
    Trading agent implementation that inherits from BaseAgent.
    This class can be extended with trading-specific functionality
    while maintaining the core agent capabilities.
    """
    def __init__(self, env: Env, ppo_params: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        # Validate parameters using utility function
        if ppo_params and not validate_trading_params(ppo_params):
            raise ValueError("Invalid trading parameters provided")
        
        super().__init__(env, ppo_params=ppo_params, seed=seed)
        self.stop_loss = DEFAULT_STOP_LOSS
        self.take_profit = DEFAULT_TAKE_PROFIT
    
    def update_state(self, portfolio_value: float, positions: Dict[str, float]) -> None:
        """
        Update agent state with new portfolio value and positions.
        Uses utility functions for validation.
        """
        # Validate portfolio value
        if not validate_numeric(portfolio_value, min_value=0):
            raise ValueError("Invalid portfolio value")
            
        # Validate position sizes
        for symbol, size in positions.items():
            if not validate_numeric(size, min_value=MIN_POSITION_SIZE, max_value=MAX_POSITION_SIZE):
                raise ValueError(f"Invalid position size for {symbol}")
                
        super().update_state(portfolio_value, positions)
