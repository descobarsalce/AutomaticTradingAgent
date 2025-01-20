"""
Trading Agent Implementation
Implements a reinforcement learning agent for automated trading using discrete actions.

The agent uses PPO (Proximal Policy Optimization) for training and supports:
- Discrete action space (buy/sell/hold)
- Portfolio state tracking
- Transaction cost handling
- Position sizing rules
"""

from typing import Dict, Any, Optional, Union, Tuple, cast
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
    """
    Trading agent implementation with discrete action support.

    Implements a trading strategy using reinforcement learning with:
    - Discrete action space (0=hold, 1=buy, 2=sell)
    - Position tracking and management
    - Risk controls including position limits and stop losses

    Attributes:
        max_position_size (float): Maximum allowed position size as % of portfolio
        min_position_size (float): Minimum allowed position size as % of portfolio
        stop_loss (float): Stop loss threshold as a decimal
    """

    def __init__(self,
                env: Env,
                ppo_params: Optional[Dict[str, Union[float, int, bool, None]]] = None,
                seed: Optional[int] = None) -> None:
        """
        Initialize trading agent with discrete actions.

        Args:
            env: Gymnasium environment for trading
            ppo_params: Configuration parameters for the PPO algorithm
            seed: Random seed for reproducibility
        """
        # Initialize base agent with policy suited for discrete actions
        super().__init__(
            env,
            ppo_params=ppo_params,
            seed=seed,
            policy_kwargs={
                'net_arch':
                dict(pi=[64, 64],
                     vf=[64, 64])  # Deeper network for discrete actions
            })

        # Trading specific parameters
        self.max_position_size = MAX_POSITION_SIZE
        self.min_position_size = MIN_POSITION_SIZE
        self.stop_loss = DEFAULT_STOP_LOSS

    def predict(self,
               observation: np.ndarray,
               deterministic: bool = True) -> np.ndarray:
        """
        Generate trading decisions based on current market observation.

        Args:
            observation: Current market state observation
            deterministic: Whether to use deterministic action selection

        Returns:
            np.ndarray: Discrete action (0=hold, 1=buy, 2=sell)

        Raises:
            ValueError: If action prediction is invalid
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
        # if action != 0:
        # logger.info(f"""
        # Agent Prediction:
        #   Raw Action: {action}
        #   Is Deterministic: {deterministic}
        #   Current Portfolio Value: {getattr(self.env, 'net_worth', 'N/A')}
        # """)
        return np.array([action])

    def update_state(self, 
                    portfolio_value: float,
                    positions: Dict[str, float]) -> None:
        """
        Update agent's internal state with current portfolio information.

        Args:
            portfolio_value: Current total portfolio value
            positions: Dictionary mapping symbols to position sizes

        Raises:
            ValueError: If position sizes are invalid or exceed limits
        """
        # Full validation only in normal mode
        for symbol, size in positions.items():
            if not isinstance(size, (int, float)) or not np.isfinite(size):
                raise ValueError(f"Invalid position size type for {symbol}")
            if abs(size) > self.max_position_size:
                raise ValueError(
                    f"Position size {size} exceeds limit for {symbol}")

        super().update_state(portfolio_value, positions)

    def train(self,
             total_timesteps: int,
             callback: Optional[BaseCallback] = None) -> None:
        """
        Train the agent on historical market data.

        Args:
            total_timesteps: Number of environment steps to train for
            callback: Optional callback for tracking training progress

        Raises:
            ValueError: If total_timesteps is not a positive integer
        """
        if not isinstance(total_timesteps, int) or total_timesteps <= 0:
            raise ValueError("total_timesteps must be a positive integer")

        # Initialize start time for progress tracking
        import time
        self.start_time = time.time()

        if callback:
            self.model.learn(total_timesteps=total_timesteps,
                             callback=callback)
        else:
            self.model.learn(total_timesteps=total_timesteps)