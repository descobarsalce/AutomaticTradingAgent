from typing import Dict, Any, Optional, Union, cast
import numpy as np
from gymnasium import Env
from stable_baselines3.common.callbacks import BaseCallback
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
        quick_mode: bool = False,
        fast_eval: bool = False
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
            # Define minimal parameters for quick training within recommended ranges
            quick_params = cast(Optional[Dict[str, Union[float, int, bool, None]]], {
                'n_steps': 512,          # Minimum recommended steps
                'batch_size': 64,        # Minimum recommended batch size
                'n_epochs': 3,           # Fewer epochs for speed
                'learning_rate': 3e-4    # Standard learning rate
            })
            ppo_params = quick_params
            
        # Use default position limits for all modes
        self.max_position_size = MAX_POSITION_SIZE
        self.min_position_size = MIN_POSITION_SIZE
        
        # Initialize base agent
        super().__init__(
            env,
            ppo_params=ppo_params,
            seed=seed,
            policy_kwargs={'net_arch': dict(pi=[32], vf=[32])} if quick_mode else None
        )
        self.stop_loss = DEFAULT_STOP_LOSS
        self.fast_eval = fast_eval
        
        # Reduce network size and skip some calculations in fast eval mode
        if fast_eval:
            self.eval_frequency = 1000  # Fixed evaluation frequency for fast eval mode
            self.skip_metrics = ['sortino_ratio', 'information_ratio']  # Skip expensive metrics

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Make a prediction and apply position size limits.
        
        Args:
            observation: Current environment observation
            deterministic: Whether to use deterministic actions
            
        Returns:
            np.ndarray: Action scaled within position limits
        """
        action = super().predict(observation, deterministic)
        
        if self.quick_mode:
            # Clip and scale actions to respect position limits
            scaled_action = np.zeros_like(action)
            for i in range(len(action)):
                # Clip action to [-1, 1] range first
                clipped_action = np.clip(action[i], -1.0, 1.0)
                # Scale based on direction while respecting limits
                if clipped_action >= 0:
                    scaled_action[i] = clipped_action * self.max_position_size * 0.95  # Add 5% buffer
                else:
                    scaled_action[i] = clipped_action * abs(self.min_position_size) * 0.95
            return scaled_action
        
        return action

    def update_state(self, portfolio_value: float, positions: Dict[str, float]) -> None:
        """
        Update agent state with portfolio information.
        Validates position sizes before updating.
        """
        # Validate position sizes
        for symbol, size in positions.items():
            if not isinstance(size, (int, float)) or not np.isfinite(size):
                raise ValueError(f"Invalid position size type for {symbol}")
            if self.quick_mode:
                if size > 0 and size > self.max_position_size:
                    raise ValueError(f"Long position size {size} exceeds quick mode limit of {self.max_position_size} for {symbol}")
                elif size < 0 and size < self.min_position_size:
                    raise ValueError(f"Short position size {size} exceeds quick mode limit of {self.min_position_size} for {symbol}")
            elif abs(size) > 10.0:  # More permissive bound for normal mode
                raise ValueError(f"Position size too large for {symbol}")
                
        super().update_state(portfolio_value, positions)
        
    def train(self, total_timesteps: int, callback: Optional[Any] = None) -> None:
        """
        Train the agent with progress tracking.

        Args:
            total_timesteps (int): Number of steps to train
            callback (Optional[BaseCallback]): Progress tracking callback for monitoring training
        """
        if not isinstance(total_timesteps, int) or total_timesteps <= 0:
            raise ValueError("total_timesteps must be a positive integer")
            
        if callback:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback
            )
        else:
            self.model.learn(total_timesteps=total_timesteps)
