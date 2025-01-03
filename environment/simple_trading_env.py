import gymnasium as gym
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SimpleTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, transaction_cost=0.0, min_transaction_size=0.001, 
                 max_position_pct=0.95, use_position_profit=True, use_holding_bonus=True, use_trading_penalty=True):
        super().__init__()
        self.use_position_profit = use_position_profit
        self.use_holding_bonus = use_holding_bonus
        self.use_trading_penalty = use_trading_penalty

        # Store data
        self.data = data

        # Initialize trading state
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.current_step = 0
        self.max_steps = len(data) if data is not None else 100

        # Track holding period and cost basis
        self.holding_period = 0
        self.cost_basis = 0
        self.last_trade_step = None

        # Transaction parameters
        self.transaction_cost = transaction_cost
        self.min_transaction_size = min_transaction_size
        self.max_position_pct = max_position_pct

        # Define action space as discrete: 0 (hold), 1 (buy), 2 (sell)
        self.action_space = gym.spaces.Discrete(3)

        # Define observation space for OHLCV + position + balance
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),  # OHLCV + position + balance
            dtype=np.float32
        )

    def _get_observation(self):
        """Get current observation of market and account state."""
        obs = np.array([
            self.data.iloc[self.current_step]['Open'],
            self.data.iloc[self.current_step]['High'],
            self.data.iloc[self.current_step]['Low'],
            self.data.iloc[self.current_step]['Close'],
            self.data.iloc[self.current_step]['Volume'],
            self.shares_held,
            self.balance
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.holding_period = 0
        self.cost_basis = 0
        self.last_trade_step = None

        observation = self._get_observation()
        info = {
            'initial_balance': self.initial_balance,
            'net_worth': self.net_worth,
            'shares_held': self.shares_held,
            'balance': self.balance
        }
        return observation, info

    def step(self, action):
        """Execute one step in the environment."""
        # Ensure action is valid
        action = int(action)
        if not 0 <= action <= 2:
            raise ValueError(f"Invalid action {action}. Must be 0 (hold), 1 (buy), or 2 (sell)")

        # Store previous state
        prev_net_worth = self.net_worth
        current_price = self.data.iloc[self.current_step]['Close']

        # Process actions
        if action == 1:  # Buy
            trade_amount = self.balance * 0.2  # Use 20% of available balance
            shares_to_buy = trade_amount / current_price
            transaction_fees = trade_amount * self.transaction_cost
            total_cost = trade_amount + transaction_fees

            if total_cost <= self.balance:
                self.balance -= total_cost
                self.shares_held += shares_to_buy
                self.cost_basis = current_price
                self.holding_period = 0
                self.last_trade_step = self.current_step

        elif action == 2:  # Sell
            if self.shares_held > 0:
                shares_to_sell = self.shares_held * 0.2  # Sell 20% of holdings
                sell_amount = shares_to_sell * current_price
                transaction_fees = sell_amount * self.transaction_cost
                net_sell_amount = sell_amount - transaction_fees

                self.balance += net_sell_amount
                self.shares_held -= shares_to_sell
                if self.shares_held == 0:
                    self.holding_period = 0
                    self.last_trade_step = None

        # Update portfolio value
        self.net_worth = self.balance + (self.shares_held * current_price)

        # Calculate position profit (used for scaling holding bonus)
        position_profit = 0
        if self.shares_held > 0 and self.cost_basis > 0:
            position_profit = (current_price - self.cost_basis) / self.cost_basis
            position_profit = np.clip(position_profit, -1, 1)

        # Update holding period for any non-zero position
        if self.shares_held > 0:
            self.holding_period += 1

        # Calculate rewards with emphasis on holding profitable positions
        reward = 0

        # 1. Base reward (minimal impact)
        if prev_net_worth > 0:
            portfolio_return = (self.net_worth - prev_net_worth) / prev_net_worth
            base_reward = portfolio_return * 0.01  # Extremely small weight (1%)
            reward += base_reward

        # 2. Position profit component (small impact)
        if self.use_position_profit and position_profit > 0:
            reward += position_profit * 0.05  # Very small weight (5%)

        # 3. Holding bonus (dominant component)
        # Guaranteed to increase linearly with time when holding
        if self.use_holding_bonus and self.shares_held > 0:
            holding_bonus = 0.5 * self.holding_period  # Large linear increase (50% per step)
            reward += holding_bonus

        # 4. Trading penalty (always makes trading worse than holding)
        if self.use_trading_penalty and action != 0:
            # Calculate current holding reward for comparison
            holding_reward = reward

            # Ensure trading reward is less than holding by scaling down significantly
            if action == 2 and self.shares_held > 0:  # Selling
                reward = holding_reward * 0.2  # Trading reward is 20% of what holding would give
            elif action == 1:  # Buying
                reward = holding_reward * 0.3  # Slightly better than selling but still much worse than holding

        # Update state
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        truncated = False

        # Get next observation
        next_observation = self._get_observation()

        # Prepare info dict
        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'holding_period': self.holding_period,
            'position_profit': position_profit,
            'reward_components': {
                'base': base_reward if 'base_reward' in locals() else 0,
                'position_profit': position_profit * 0.05 if self.use_position_profit else 0,
                'holding_bonus': holding_bonus if 'holding_bonus' in locals() else 0,
                'trading_penalty': 'applied' if self.use_trading_penalty and action != 0 else 'none'
            }
        }

        return next_observation, reward, done, truncated, info