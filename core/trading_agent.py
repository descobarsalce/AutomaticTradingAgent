"""Trading Agent Implementation"""

from typing import Dict, Any, Optional, Union, Tuple, cast, List, Callable
import numpy as np
import logging
from gymnasium import Env
from stable_baselines3.common.callbacks import BaseCallback
from core.base_agent import BaseAgent
from utils.data_utils import validate_numeric
from utils.common import (MAX_POSITION_SIZE, MIN_POSITION_SIZE,
                        DEFAULT_STOP_LOSS, validate_trading_params)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TradingAgent(BaseAgent):
    """Trading agent implementation using PPO algorithm for discrete action trading."""

    def __init__(self, env: Env,
                ppo_params: Optional[Dict[str, Union[float, int, bool, None]]] = None,
                seed: Optional[int] = None) -> None:
        """Initialize trading agent with discrete actions."""
        super().__init__(
            env,
            ppo_params=ppo_params,
            seed=seed,
            policy_kwargs={
                'net_arch': dict(pi=[64, 64], vf=[64, 64])
            })

        self.max_position_size = MAX_POSITION_SIZE
        self.min_position_size = MIN_POSITION_SIZE
        self.stop_loss = DEFAULT_STOP_LOSS

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Generate trading decisions (0=hold, 1=buy, 2=sell)."""
        action = super().predict(observation, deterministic)

        if not isinstance(action, (np.ndarray, int)):
            raise ValueError("Invalid action type")

        action = int(action)
        if not 0 <= action <= 2:
            raise ValueError(f"Invalid action value: {action}")

        return np.array([action])

    def update_state(self, portfolio_value: float, positions: Dict[str, float]) -> None:
        """Update agent's state with current portfolio information."""
        for symbol, size in positions.items():
            if not isinstance(size, (int, float)) or not np.isfinite(size):
                raise ValueError(f"Invalid position size type for {symbol}")
            if abs(size) > self.max_position_size:
                raise ValueError(f"Position size {size} exceeds limit for {symbol}")

        super().update_state(portfolio_value, positions)

    def train(self, total_timesteps: int, callback: Optional[BaseCallback] = None) -> None:
        """Train the agent on historical market data."""
        if not isinstance(total_timesteps, int) or total_timesteps <= 0:
            raise ValueError("total_timesteps must be a positive integer")

        import time
        self.start_time = time.time()

        if callback:
            self.model.learn(total_timesteps=total_timesteps, callback=callback)
        else:
            self.model.learn(total_timesteps=total_timesteps)