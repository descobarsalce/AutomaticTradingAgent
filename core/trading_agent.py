from typing import Dict, Any, Optional, Union
import numpy as np
from gymnasium import Env
from .base_agent import BaseAgent
from utils.data_utils import validate_numeric
from utils.common import (
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
    def __init__(self, env: Env, ppo_params: Optional[Dict[str, Any]] = None, 
                seed: Optional[int] = None, quick_mode: bool = False):
        """
        Initialize the trading agent.
        
        Args:
            env: The trading environment
            ppo_params: Optional PPO parameters dictionary
            seed: Optional random seed
            quick_mode: If True, use minimal parameters for fast testing
        """
        # Set quick mode parameters if enabled
        if quick_mode:
            quick_params = {
                'learning_rate': 3e-4,      # Standard learning rate
                'n_steps': 512,             # Minimum recommended steps
                'batch_size': 64,           # Minimum recommended batch size
                'n_epochs': 5,              # Fewer epochs
                'gamma': 0.95,              # Slightly lower discount factor
                'clip_range': 0.2,          # Standard clip range
                'ent_coef': 0.01,           # Standard entropy coefficient
                'policy_kwargs': {
                    'net_arch': {'pi': [32, 32], 'vf': [32, 32]},  # Smaller network
                    'activation_fn': 'tanh'  # Better for bounded actions
                }
            }
            ppo_params = quick_params
        
        # Validate parameters using utility function
        if ppo_params and not validate_trading_params(ppo_params):
            raise ValueError("Invalid trading parameters provided")
        
        super().__init__(env, ppo_params=ppo_params, seed=seed)
        self.stop_loss = DEFAULT_STOP_LOSS
        self.take_profit = DEFAULT_TAKE_PROFIT
        self.quick_mode = quick_mode
    
    def update_state(self, portfolio_value: float, positions: Dict[str, float]) -> None:
        """
        Update agent state with new portfolio value and positions.
        Uses utility functions for validation with more permissive checks for training.
        """
        # Validate portfolio value
        if not validate_numeric(portfolio_value, min_value=0):
            raise ValueError("Invalid portfolio value")
            
        # Validate position sizes with more permissive checks
        for symbol, size in positions.items():
            # Allow any finite number within reasonable bounds (-10 to 10)
            if not isinstance(size, (int, float)) or not np.isfinite(size):
                raise ValueError(f"Invalid position size type for {symbol}")
            if abs(size) > 10.0:  # More permissive bound for training
                raise ValueError(f"Position size too large for {symbol}")
                
        super().update_state(portfolio_value, positions)
