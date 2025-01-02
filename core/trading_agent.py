from typing import Dict, Any, Optional, Union, cast
import numpy as np
from gymnasium import Env
from stable_baselines3.common.callbacks import BaseCallback
from core.base_agent import BaseAgent
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
                'n_steps': 256,          # Reduced steps for quick training
                'batch_size': 128,       # Balanced batch size
                'n_epochs': 3,           # Fewer epochs for speed
                'learning_rate': 3e-4,   # Optimal learning rate
                'clip_range': 0.2,       # Less aggressive clipping
                'target_kl': 0.05,       # Relaxed KL divergence target
                'use_sde': False         # Disable SDE for simpler training
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
        
        # Optimize for fast evaluation mode
        if fast_eval:
            self.eval_frequency = 5000  # Reduced evaluation frequency
            self.skip_metrics = ['sortino_ratio', 'information_ratio', 'sharpe_ratio']  # Skip all expensive metrics
            self.quick_validation = True  # Enable quick validation
            # Use smaller network for faster inference
            self.policy_kwargs = {'net_arch': dict(pi=[32], vf=[32])}

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
            # Scale actions to ensure they meet minimum transaction requirements
            scaled_action = np.zeros_like(action)
            for i in range(len(action)):
                clipped_action = np.clip(action[i], -1.0, 1.0)
                
                # Ensure action magnitude meets minimum transaction size
                if abs(clipped_action) < 0.1:  # Dead zone for very small actions
                    scaled_action[i] = 0
                else:
                    # Scale to position size while ensuring minimum transaction size
                    if clipped_action >= 0:
                        scaled_action[i] = max(0.2, clipped_action) * self.max_position_size
                    else:
                        scaled_action[i] = min(-0.2, clipped_action) * abs(self.min_position_size)
            return scaled_action
        
        return action

    def update_state(self, portfolio_value: float, positions: Dict[str, float]) -> None:
        """
        Update agent state with portfolio information.
        Uses simplified validation in fast eval mode.
        """
        if not self.fast_eval:
            # Full validation only in normal mode
            for symbol, size in positions.items():
                if not isinstance(size, (int, float)) or not np.isfinite(size):
                    raise ValueError(f"Invalid position size type for {symbol}")
                if abs(size) > self.max_position_size:
                    raise ValueError(f"Position size {size} exceeds limit for {symbol}")
        
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

        # Initialize start time for progress tracking
        import time
        self.start_time = time.time()
            
        if callback:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback
            )
        else:
            if self.quick_mode or self.fast_eval:
                # Simple progress reporting for quick/fast modes with reduced timesteps
                reduced_steps = min(total_timesteps // 4, 1000) if self.fast_eval else total_timesteps
                print(f"Starting {'quick' if self.quick_mode else 'fast'} training...")
                self.model.learn(total_timesteps=reduced_steps)
                print("Training completed")
            else:
                self.model.learn(total_timesteps=total_timesteps)