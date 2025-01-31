"""Trading environment with continuous action space for portfolio allocation"""
import gymnasium as gym
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from gymnasium import spaces

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TradingEnv(gym.Env):
    """
    Trading environment that implements continuous action space for portfolio allocation:
    - Actions represent portfolio weights (0-1) for each asset
    - Supports simultaneous positions across multiple assets
    - Implements proper portfolio rebalancing
    """
    def __init__(self,
                 data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 initial_balance: float = 10000,
                 transaction_cost: float = 0.001,
                 window_size: int = 30,
                 training_mode: bool = False):
        super().__init__()

        # Portfolio tracking
        self._portfolio_history = []
        self._portfolio_weights = []
        self.training_mode = training_mode
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        # Handle both single and multi-asset data
        self.data = data if isinstance(data, dict) else {'asset': data}
        self.symbols = list(self.data.keys())
        self.num_assets = len(self.symbols)

        # Continuous action space: portfolio weights for each asset
        # Each weight is between 0 and 1, sum of weights <= 1
        self.action_space = spaces.Box(
            low=0, high=1, 
            shape=(self.num_assets,), 
            dtype=np.float32
        )

        # Observation space includes price data, technical indicators, and current positions
        features_per_asset = 5  # OHLCV
        obs_dim = (self.num_assets * features_per_asset) + self.num_assets + 1  # +positions +balance
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.reset_portfolio_and_balance()

    def _get_observation(self) -> np.ndarray:
        """Get current observation of market and account state."""
        obs = []
        for symbol in self.symbols:
            data = self.data[symbol].iloc[self.current_step]
            # Add OHLCV data
            obs.extend([
                float(data['Open']),
                float(data['High']), 
                float(data['Low']),
                float(data['Close']),
                float(data['Volume'])
            ])
        # Add current positions
        obs.extend([float(self.positions[symbol]) for symbol in self.symbols])
        # Add account balance
        obs.append(float(self.balance))

        return np.array(obs, dtype=np.float32)

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize portfolio weights to sum to 1.0"""
        weights = np.clip(weights, 0, 1)
        if np.sum(weights) > 0:
            return weights / np.sum(weights)
        return weights

    def reset_portfolio_and_balance(self) -> Tuple[np.ndarray, Dict]:
        """Reset the portfolio and balance to initial state."""
        self._portfolio_history = []
        self._portfolio_weights = []
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.portfolio_value = self.initial_balance

        observation = self._get_observation()
        info = {
            'initial_balance': self.initial_balance,
            'portfolio_value': self.portfolio_value,
            'positions': self.positions.copy(),
            'portfolio_weights': np.zeros(self.num_assets),
            'balance': self.balance
        }
        return observation, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        return self.reset_portfolio_and_balance()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment using continuous portfolio weights."""
        self.current_step += 1

        # Store previous portfolio value for reward calculation
        prev_portfolio_value = self.portfolio_value

        # Normalize action weights
        target_weights = self._normalize_weights(action)

        # Calculate current portfolio weights
        total_value = self.balance + sum(
            self.positions[symbol] * self.data[symbol].iloc[self.current_step]['Close']
            for symbol in self.symbols
        )

        current_weights = np.array([
            self.positions[symbol] * self.data[symbol].iloc[self.current_step]['Close'] / total_value
            if total_value > 0 else 0.0
            for symbol in self.symbols
        ])

        # Rebalance portfolio to match target weights
        for i, symbol in enumerate(self.symbols):
            current_price = self.data[symbol].iloc[self.current_step]['Close']
            target_value = total_value * target_weights[i]
            current_value = self.positions[symbol] * current_price

            if target_value > current_value:  # Need to buy
                shares_to_buy = (target_value - current_value) / current_price
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.positions[symbol] += shares_to_buy
                    self.balance -= cost

            elif target_value < current_value:  # Need to sell
                shares_to_sell = (current_value - target_value) / current_price
                revenue = shares_to_sell * current_price * (1 - self.transaction_cost)
                self.positions[symbol] -= shares_to_sell
                self.balance += revenue

        # Update portfolio value
        self.portfolio_value = self.balance + sum(
            self.positions[symbol] * self.data[symbol].iloc[self.current_step]['Close']
            for symbol in self.symbols
        )
        self._portfolio_history.append(self.portfolio_value)
        self._portfolio_weights.append(target_weights)

        # Calculate reward (Sharpe Ratio approximation)
        returns = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        reward = returns

        # Get next observation
        observation = self._get_observation()

        # Check if episode is done
        done = self.current_step >= len(next(iter(self.data.values()))) - 1
        truncated = False

        # Prepare info dict
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'portfolio_weights': target_weights,
            'date': self.data[self.symbols[0]].iloc[self.current_step].name,
            'returns': returns
        }

        return observation, reward, done, truncated, info

    def get_portfolio_history(self) -> Tuple[List[float], List[np.ndarray]]:
        """Return the history of portfolio values and weights."""
        return self._portfolio_history, self._portfolio_weights