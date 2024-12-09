from typing import Dict, Any, Optional, Union
import numpy as np
from gymnasium import Env
from .base_agent import BaseAgent
from utils.data_utils import validate_numeric
from utils.common import (
    MAX_POSITION_SIZE,
    MIN_POSITION_SIZE,
    DEFAULT_STOP_LOSS,
    validate_trading_params
)

class TradingAgent(BaseAgent):
    """Trading agent with quick training mode capabilities."""
    
    def __init__(
        self,
        env: Env,
        ppo_params: Optional[Dict[str, Union[float, int, bool, None]]] = None,
        seed: Optional[int] = None,
        quick_mode: bool = False
    ) -> None:
        """
        Initialize trading agent with optional quick mode.
        
        Args:
            env: Trading environment
            ppo_params: Optional PPO parameters
            seed: Optional random seed
            quick_mode: If True, use minimal parameters for fast testing
        """
        # Store quick mode setting
        self.quick_mode = quick_mode
        
        # Set quick mode parameters if enabled
        if quick_mode:
            # Define minimal parameters for quick training
            quick_params = {
                'n_steps': 512,          # Minimum recommended steps
                'batch_size': 64,        # Minimum recommended batch size
                'n_epochs': 3,           # Fewer epochs for speed
                'learning_rate': 3e-4,   # Standard learning rate
                'clip_range': 0.1,       # Smaller clip range for stability
                'ent_coef': 0.005,       # Lower entropy for focused learning
                'policy_kwargs': {        # Smaller network architecture
                    'net_arch': dict(pi=[32], vf=[32])
                }
            }
            ppo_params = quick_params
            
            # Set more conservative position limits for quick mode
            self.max_position_size = 0.2  # Limit position size to 20% of portfolio
            self.min_position_size = -0.2  # Limit short positions to 20% of portfolio
        
        # Validate parameters using utility function
        if ppo_params and not validate_trading_params(ppo_params):
            raise ValueError("Invalid trading parameters provided")
        
        # Initialize base agent
        super().__init__(
            env,
            ppo_params=ppo_params,
            seed=seed
        )
        self.stop_loss = DEFAULT_STOP_LOSS

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Make a prediction and apply position size limits in quick mode.
        
        Args:
            observation: Current environment observation
            deterministic: Whether to use deterministic actions
            
        Returns:
            np.ndarray: Action scaled within position limits
        """
        action = super().predict(observation, deterministic)
        
        if self.quick_mode:
            # Scale action to be within quick mode limits while preserving direction
            position_range = self.max_position_size - self.min_position_size
            scaled_action = (action + 1) * position_range / 2 + self.min_position_size
            return np.clip(scaled_action, self.min_position_size, self.max_position_size)
        
        return action

    def update_state(self, portfolio_value: float, positions: Dict[str, float]) -> None:
        """
        Update agent state with portfolio information.
        Validates position sizes before updating.
        """
        # Validate position sizes
        for symbol, size in positions.items():
            # Allow any finite number within reasonable bounds (-10 to 10)
            if not isinstance(size, (int, float)) or not np.isfinite(size):
                raise ValueError(f"Invalid position size type for {symbol}")
            if abs(size) > 10.0:  # More permissive bound for training
                raise ValueError(f"Position size too large for {symbol}")
                
        super().update_state(portfolio_value, positions)
