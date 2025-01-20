"""
Trading Agent Implementation
Implements a reinforcement learning agent for automated trading using discrete actions.

System Architecture:
1. Policy Network:
   - Actor-Critic architecture using PPO
   - Discrete action space mapping
   - State preprocessing

2. Position Management:
   - Position size limits
   - Stop loss implementation
   - Transaction cost handling

3. Risk Controls:
   - Maximum position limits
   - Stop loss mechanisms
   - Exposure tracking

4. Training Pipeline:
   - Experience collection
   - Policy optimization
   - Performance monitoring

Key Features:
- PPO (Proximal Policy Optimization) implementation
- Discrete action space (buy/sell/hold)
- Portfolio state tracking
- Transaction cost modeling
- Risk management rules
- Position sizing logic

Component Interactions:
1. Environment Interface:
   - Receives market observations
   - Executes trading decisions
   - Tracks portfolio state

2. Model Management:
   - Policy network updates
   - Experience replay
   - State normalization

3. Risk Controls:
   - Position limit enforcement
   - Stop loss triggering
   - Exposure calculation

Example Usage:
```python
# Initialize agent
env = TradingEnvironment(data)
agent = TradingAgent(
    env=env,
    ppo_params={
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64
    }
)

# Training
agent.train(total_timesteps=100000)

# Trading
observation = env.reset()
action = agent.predict(observation)
```
"""

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
    """
    Trading agent implementation with discrete action support.

    Implementation Architecture:
    1. State Processing:
       - Market data normalization
       - Position encoding
       - Account state tracking

    2. Action Generation:
       - Policy network forward pass
       - Action space mapping
       - Position sizing rules

    3. Risk Management:
       - Position limit enforcement
       - Stop loss checking
       - Exposure tracking

    4. Training Loop:
       - Experience collection
       - Policy optimization
       - Performance monitoring

    Attributes:
        max_position_size (float): Maximum allowed position size as % of portfolio
        min_position_size (float): Minimum allowed position size as % of portfolio
        stop_loss (float): Stop loss threshold as a decimal
        env (Env): Trading environment instance
        model (PPO): Policy network model
        training_started (bool): Training status flag

    Implementation Notes:
        - Uses PPO algorithm for policy optimization
        - Implements discrete action space (0=hold, 1=buy, 2=sell)
        - Enforces position limits and risk controls
        - Tracks portfolio state and performance metrics

    Example:
        ```python
        # Initialize agent
        env = TradingEnvironment(data)
        agent = TradingAgent(
            env=env,
            ppo_params={'learning_rate': 3e-4}
        )

        # Training phase
        agent.train(total_timesteps=100000)

        # Trading phase
        obs = env.reset()
        while not done:
            action = agent.predict(obs)
            obs, reward, done, info = env.step(action)
        ```
    """

    def __init__(self,
                env: Env,
                ppo_params: Optional[Dict[str, Union[float, int, bool, None]]] = None,
                seed: Optional[int] = None) -> None:
        """
        Initialize trading agent with discrete actions.

        Implementation Steps:
        1. Environment Setup:
           - Validate environment interface
           - Initialize action/observation spaces

        2. Policy Configuration:
           - Setup neural network architecture
           - Configure PPO parameters
           - Initialize optimizer

        3. State Management:
           - Setup position tracking
           - Initialize risk controls
           - Configure logging

        Args:
            env (Env): Gymnasium environment for trading with:
                - action_space: Discrete(3)
                - observation_space: Box(shape=(n_features,))
            ppo_params (Optional[Dict[str, Union[float, int, bool, None]]]): 
                Configuration parameters for PPO algorithm including:
                - learning_rate: float - Policy learning rate
                - n_steps: int - Steps per update
                - batch_size: int - Training batch size
                - n_epochs: int - Number of training epochs
                - gamma: float - Discount factor
                - gae_lambda: float - GAE parameter
                - clip_range: float - PPO clip range
            seed (Optional[int]): Random seed for reproducibility

        Raises:
            ValueError: If environment spaces are incompatible
            TypeError: If parameter types are invalid

        Example:
            ```python
            env = TradingEnvironment(data)
            agent = TradingAgent(
                env=env,
                ppo_params={
                    'learning_rate': 3e-4,
                    'n_steps': 2048,
                    'batch_size': 64,
                    'n_epochs': 10,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_range': 0.2
                }
            )
            ```
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

        Implementation Process:
        1. State Processing:
           - Validate observation format
           - Apply feature normalization
           - Encode position information

        2. Action Generation:
           - Forward pass through policy network
           - Action space mapping
           - Deterministic vs stochastic selection

        3. Risk Validation:
           - Check position limits
           - Validate action bounds
           - Log prediction details

        Args:
            observation (np.ndarray): Current market state observation containing:
                - Price features (OHLCV data)
                - Technical indicators
                - Position information
                Shape: (n_features,) where n_features matches env.observation_space

            deterministic (bool): Whether to use deterministic action selection
                - True: Use mode of policy distribution (default)
                - False: Sample from policy distribution for exploration

        Returns:
            np.ndarray: Discrete action vector where:
                - 0: Hold current position
                - 1: Buy/increase position
                - 2: Sell/decrease position
                Shape: (1,) containing single integer action

        Raises:
            ValueError: If action prediction is invalid
            TypeError: If observation has incorrect type/shape

        Example:
            ```python
            # Get market observation
            observation = env.get_observation()

            # Generate trading decision
            action = agent.predict(
                observation=observation,
                deterministic=True  # Use deterministic policy
            )

            # Execute trade
            next_obs, reward, done, info = env.step(action)
            ```
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

        Implementation Process:
        1. State Validation:
           - Check value types
           - Validate position sizes
           - Verify portfolio value

        2. Risk Assessment:
           - Calculate total exposure
           - Check position limits
           - Validate stop losses

        3. State Update:
           - Update position tracking
           - Record portfolio value
           - Update risk metrics

        Args:
            portfolio_value (float): Current total portfolio value including:
                - Cash balance
                - Position market values
                - Unrealized P&L
                Must be positive and finite

            positions (Dict[str, float]): Dictionary mapping symbols to position sizes
                - Keys: Symbol identifiers (str)
                - Values: Position sizes as % of portfolio (float)
                Example: {'AAPL': 0.5, 'GOOGL': -0.3}

        Raises:
            ValueError: If position sizes are invalid or exceed limits
            TypeError: If input types are incorrect

        Example:
            ```python
            # Update agent state
            agent.update_state(
                portfolio_value=100000.0,
                positions={
                    'AAPL': 0.5,  # 50% long AAPL
                    'GOOGL': -0.3  # 30% short GOOGL
                }
            )
            ```
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

        Implementation Process:
        1. Training Setup:
           - Validate parameters
           - Initialize progress tracking
           - Setup experience buffer

        2. Training Loop:
           - Collect experience batches
           - Optimize policy network
           - Track performance metrics

        3. Progress Monitoring:
           - Update callback status
           - Log training metrics
           - Save checkpoints

        Args:
            total_timesteps (int): Number of environment steps to train for
                - Must be positive integer
                - Typically 100k-1M steps for convergence
                - Affects training duration and stability

            callback (Optional[BaseCallback]): Optional callback for tracking progress
                - Reports training metrics
                - Updates progress bars
                - Handles early stopping
                Must implement BaseCallback interface

        Raises:
            ValueError: If total_timesteps is not a positive integer
            RuntimeError: If training fails to converge

        Example:
            ```python
            # Setup training callback
            callback = ProgressCallback(
                check_freq=1000,
                verbose=1
            )

            # Train the agent
            agent.train(
                total_timesteps=100000,
                callback=callback
            )
            ```

        Notes:
            - Training duration depends on total_timesteps and hardware
            - Higher timesteps generally improve policy quality
            - Use callback to monitor convergence
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