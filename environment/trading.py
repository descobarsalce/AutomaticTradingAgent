import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd

class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=100000):
        super(TradingEnvironment, self).__init__()

        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0

        # Action space: discrete values (0: hold, 1: buy, 2: sell)
        self.action_space = spaces.Discrete(3)

        # Observation space: price data + account info
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),  # OHLCV + position + balance
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)  # Reset the base env with seed
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.cost_basis = 0
        self.holding_period = 0  # Track holding period for reward shaping

        observation = self._get_observation()
        info = {
            'initial_balance': self.initial_balance,
            'net_worth': self.net_worth,
            'shares_held': self.shares_held,
            'holding_period': self.holding_period
        }
        return observation, info

    def _get_observation(self):
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

    def _step_impl(self, action):
        """Core step logic implementation with discrete actions."""
        # Validate discrete action
        action = int(action)
        if not 0 <= action <= 2:
            raise ValueError(f"Invalid action {action}. Must be 0 (hold), 1 (buy), or 2 (sell)")

        current_price = self.data.iloc[self.current_step]['Close']
        prev_net_worth = self.balance + self.shares_held * current_price  # Calculate true previous net worth

        # Fixed position sizing (20% of balance for each trade)
        trade_amount = self.balance * 0.2

        if action == 1:  # Buy
            max_shares = trade_amount / current_price
            shares_to_buy = min(max_shares, trade_amount / current_price)
            cost = shares_to_buy * current_price
            
            if cost <= self.balance:
                self.balance -= cost
                self.shares_held += shares_to_buy
                self.cost_basis = current_price
                self.holding_period = 0

        elif action == 2:  # Sell
            if self.shares_held > 0:
                shares_to_sell = min(self.shares_held, self.shares_held * 0.2)
                revenue = shares_to_sell * current_price
                self.balance += revenue
                self.shares_held -= shares_to_sell
                if self.shares_held == 0:
                    self.holding_period = 0

        # Update holding period for non-zero positions
        if self.shares_held > 0:
            self.holding_period += 1

        # Update portfolio value with current price
        self.net_worth = self.balance + (self.shares_held * current_price)

        # Base reward from portfolio return
        if prev_net_worth > 0:
            reward = ((self.net_worth - prev_net_worth) / prev_net_worth) * 0.1  # Scaled down base reward
        else:
            reward = 0

        # Add holding bonus for profitable positions
        if self.shares_held > 0 and current_price > self.cost_basis:
            holding_bonus = 0.5 * self.holding_period  # Large linear increase
            reward += holding_bonus

        # Apply trading penalty
        if action != 0:  # Penalty for active trading
            reward *= 0.5  # 50% penalty for trading vs holding

        # Update state
        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False

        obs = self._get_observation()
        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'holding_period': self.holding_period
        }

        return obs, reward, terminated, truncated, info

    def step(self, action):
        """Execute one step in the environment using discrete action value."""
        return self._step_impl(action)