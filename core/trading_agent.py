from typing import Dict, Any, Optional, Union, cast
import numpy as np
import logging
from gymnasium import Env
from stable_baselines3.common.callbacks import BaseCallback

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from core.base_agent import BaseAgent
from utils.data_utils import validate_numeric
from utils.common import (
    MAX_POSITION_SIZE,
    MIN_POSITION_SIZE,
    DEFAULT_STOP_LOSS,
    validate_trading_params
)

class TradingAgent(BaseAgent):
    """Trading agent with discrete action support."""

    def __init__(
        self,
        env: Env,
        ppo_params: Optional[Dict[str, Union[float, int, bool, None]]] = None,
        seed: Optional[int] = None,
        quick_mode: bool = False,
        fast_eval: bool = False
    ) -> None:
        """Initialize trading agent with discrete actions."""
        # Store quick mode setting
        self.quick_mode = quick_mode

        # Set quick mode parameters if enabled
        if quick_mode:
            quick_params = cast(Optional[Dict[str, Union[float, int, bool, None]]], {
                'n_steps': 128,          # Reduced steps for quick training
                'batch_size': 32,        # Smaller batch size for speed
                'n_epochs': 3,           # Fewer epochs
                'learning_rate': 3e-4,   # Standard learning rate
                'clip_range': 0.2,       # Standard clipping
                'target_kl': 0.05,       # Standard KL target
                'use_sde': False         # Disable SDE for discrete actions
            })
            ppo_params = quick_params

        # Initialize base agent with policy suited for discrete actions
        super().__init__(
            env,
            ppo_params=ppo_params,
            seed=seed,
            policy_kwargs={
                'net_arch': dict(pi=[64, 64], vf=[64, 64])  # Deeper network for discrete actions
            } if not quick_mode else dict(pi=[32], vf=[32])
        )

        # Trading specific parameters
        self.max_position_size = MAX_POSITION_SIZE
        self.min_position_size = MIN_POSITION_SIZE
        self.stop_loss = DEFAULT_STOP_LOSS
        self.fast_eval = fast_eval

        # Optimize for fast evaluation mode
        if fast_eval:
            self.eval_frequency = 5000
            self.skip_metrics = ['sortino_ratio', 'information_ratio']
            self.quick_validation = True

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Make a prediction using discrete actions.

        Args:
            observation: Current environment observation
            deterministic: Whether to use deterministic actions

        Returns:
            np.ndarray: Discrete action (0=hold, 1=buy, 2=sell)
        """
        # Get raw action from model
        action = super().predict(observation, deterministic)

        # Ensure action is discrete and valid
        if not isinstance(action, (np.ndarray, int)):
            raise ValueError("Invalid action type")

        # Convert to integer action
        action = int(action)
        if not 0 <= action <= 2:
            raise ValueError(f"Invalid action value: {action}")

        # Log prediction details
        logger.info(f"""
Agent Prediction:
  Raw Action: {action}
  Is Deterministic: {deterministic}
  Current Portfolio Value: {getattr(self.env, 'net_worth', 'N/A')}
""")
        return np.array([action])

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

    def train(self, total_timesteps: int, callback: Optional[BaseCallback] = None) -> None:
        """
        Train the agent with progress tracking.

        Args:
            total_timesteps (int): Number of steps to train
            callback (Optional[BaseCallback]): Progress tracking callback
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
                # Simple progress reporting for quick/fast modes
                reduced_steps = min(total_timesteps // 4, 1000) if self.fast_eval else total_timesteps
                print(f"Starting {'quick' if self.quick_mode else 'fast'} training...")
                self.model.learn(total_timesteps=reduced_steps)
                print("Training completed")
            else:
                self.model.learn(total_timesteps=total_timesteps)