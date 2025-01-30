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
    Trading environment with continuous action space for portfolio allocation.
    Actions represent the target portfolio weights for each asset.
    """
    def __init__(self,
                 data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 initial_balance: float = 10000,
                 transaction_cost: float = 0.0,
                 use_position_profit: bool = False,
                 use_holding_bonus: bool = False,
                 use_trading_penalty: bool = False,
                 training_mode: bool = True,
                 log_frequency: int = 30):
        super().__init__()

        self._portfolio_history = []
        self.use_position_profit = use_position_profit
        self.use_holding_bonus = use_holding_bonus
        self.use_trading_penalty = use_trading_penalty
        self.training_mode = training_mode
        self.log_frequency = log_frequency

        # Handle both single and multi-asset data
        self.data = data if isinstance(data, dict) else {'asset': data}
        self.symbols = list(self.data.keys())

        # Continuous action space: portfolio weights for each asset
        # Each weight is between 0 and 1, representing the portion of portfolio to allocate
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(self.symbols),),
            dtype=np.float32
        )

        # Observation space includes OHLCV + positions + balance for each asset
        obs_dim = (len(self.symbols) * 6) + 1  # OHLCV + position for each asset + balance
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Initialize trading state
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.cost_bases = {symbol: 0.0 for symbol in self.symbols}
        self.holding_periods = {symbol: 0 for symbol in self.symbols}
        self.net_worth = initial_balance
        self.current_step = 0
        self.last_logged_step = -1
        self.episode_count = 0
        self.total_steps = 0
        self.episode_trades = {symbol: 0 for symbol in self.symbols}
        self.transaction_cost = transaction_cost

    def _get_observation(self) -> np.ndarray:
        """Get current observation of market and account state."""
        obs = []
        for symbol in self.symbols:
            data = self.data[symbol].iloc[self.current_step]
            obs.extend([
                float(data['Open']),
                float(data['High']), 
                float(data['Low']),
                float(data['Close']),
                float(data['Volume']),
                float(self.positions[symbol])
            ])
        obs.append(float(self.balance))
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, prev_net_worth: float, current_net_worth: float,
                       actions: np.ndarray, trades_executed: Dict[str, bool]) -> float:
        """Calculate reward based on trading performance and behavior."""
        reward = 0.0

        # Portfolio return reward
        if prev_net_worth > 0:
            portfolio_return = (current_net_worth - prev_net_worth) / prev_net_worth
            reward += portfolio_return

        if self.use_holding_bonus:
            # Add holding bonus for profitable positions
            for symbol in self.symbols:
                current_price = self.data[symbol].iloc[self.current_step]['Close']
                if self.positions[symbol] > 0 and current_price > self.cost_bases[symbol]:
                    reward += 0.001 * self.holding_periods[symbol]

        if self.use_trading_penalty:
            # Penalize excessive trading
            trade_penalty = sum(trades_executed.values()) * 0.0001
            reward -= trade_penalty

        return reward

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment using continuous portfolio weights."""
        self.total_steps += 1

        # Ensure action is numpy array and normalize weights to sum to 1
        weights = np.array(action, dtype=np.float32)
        weights = np.clip(weights, 0, 1)
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight

        # Store previous state
        prev_net_worth = self.net_worth
        trades_executed = {symbol: False for symbol in self.symbols}

        # Calculate target positions based on weights
        current_prices = {symbol: self.data[symbol].iloc[self.current_step]['Close']
                         for symbol in self.symbols}
        target_positions = {}

        # First pass: calculate theoretical target positions
        for idx, symbol in enumerate(self.symbols):
            target_value = self.net_worth * weights[idx]
            target_positions[symbol] = target_value / current_prices[symbol]

        # Second pass: execute trades to reach target positions
        for idx, symbol in enumerate(self.symbols):
            current_position = self.positions[symbol]
            target_position = target_positions[symbol]
            current_price = current_prices[symbol]

            # Calculate position difference
            position_diff = target_position - current_position

            if abs(position_diff) > 1e-6:  # Threshold to avoid tiny trades
                if position_diff > 0:  # Need to buy
                    cost = abs(position_diff * current_price)
                    total_cost = cost + self.transaction_cost

                    if total_cost <= self.balance:
                        self.balance -= total_cost
                        if current_position > 0:
                            # Update cost basis for additional purchase
                            old_cost = self.cost_bases[symbol] * current_position
                            new_cost = current_price * position_diff
                            self.cost_bases[symbol] = (old_cost + new_cost) / target_position
                        else:
                            self.cost_bases[symbol] = current_price

                        self.positions[symbol] = target_position
                        trades_executed[symbol] = True
                        logger.info(f"BUY  | {symbol:5} | Price: ${current_price:.2f} | Shares: {position_diff:.4f}")

                elif position_diff < 0:  # Need to sell
                    sell_amount = abs(position_diff * current_price)
                    net_sell_amount = sell_amount - self.transaction_cost

                    self.balance += net_sell_amount
                    self.positions[symbol] = target_position
                    trades_executed[symbol] = True
                    logger.info(f"SELL | {symbol:5} | Price: ${current_price:.2f} | Shares: {abs(position_diff):.4f}")

            # Update holding period
            if self.positions[symbol] > 0:
                self.holding_periods[symbol] += 1
            else:
                self.holding_periods[symbol] = 0

        # Update portfolio value
        self.net_worth = self.balance + sum(
            self.positions[symbol] * current_prices[symbol]
            for symbol in self.symbols
        )
        self._portfolio_history.append(self.net_worth)

        # Calculate reward
        reward = self._compute_reward(prev_net_worth, self.net_worth, weights, trades_executed)

        # Update state
        self.current_step += 1
        done = self.current_step >= len(next(iter(self.data.values()))) - 1
        truncated = False

        # Get next observation
        observation = self._get_observation()

        # Prepare info dict
        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'trades_executed': trades_executed,
            'episode_trades': self.episode_trades.copy(),
            'portfolio_weights': weights,
            'date': self.data[self.symbols[0]].iloc[self.current_step].name
        }
        self.last_info = info

        return observation, reward, done, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.reset_portfolio_and_balance()

        observation = self._get_observation()
        info = {
            'initial_balance': self.initial_balance,
            'net_worth': self.net_worth,
            'positions': self.positions.copy(),
            'balance': self.balance,
            'episode': self.episode_count,
            'total_steps': self.total_steps
        }
        return observation, info

    def reset_portfolio_and_balance(self) -> None:
        """Reset the portfolio and balance to initial state."""
        self._portfolio_history = []
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.cost_bases = {symbol: 0.0 for symbol in self.symbols}
        self.holding_periods = {symbol: 0 for symbol in self.symbols}
        self.net_worth = self.initial_balance
        self.last_logged_step = -1
        self.episode_trades = {symbol: 0 for symbol in self.symbols}
        self.episode_count += 1

    def get_portfolio_history(self) -> List[float]:
        """Return the history of portfolio values."""
        return self._portfolio_history