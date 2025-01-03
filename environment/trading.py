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

        observation = self._get_observation()
        info = {
            'initial_balance': self.initial_balance,
            'net_worth': self.net_worth,
            'shares_held': self.shares_held
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
        current_price = self.data.iloc[self.current_step]['Close']

        # Fixed position sizing (20% of balance for each trade)
        trade_amount = self.balance * 0.2

        if action == 1:  # Buy
            shares_bought = trade_amount / current_price
            if shares_bought * current_price <= self.balance:  # Check sufficient balance
                self.balance -= shares_bought * current_price
                self.shares_held += shares_bought
                self.cost_basis = current_price

        elif action == 2:  # Sell
            if self.shares_held > 0:  # Only sell if we have shares
                shares_sold = min(self.shares_held, self.shares_held * 0.2)  # Sell 20% of holdings
                self.balance += shares_sold * current_price
                self.shares_held -= shares_sold

        # action == 0 is hold, no action needed

        # Calculate reward (consider both profit and risk)
        self.net_worth = self.balance + self.shares_held * current_price
        reward = (self.net_worth - self.initial_balance) / self.initial_balance

        # Update state
        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False  # Required for gymnasium compatibility

        obs = self._get_observation()

        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price
        }

        return obs, reward, terminated, truncated, info

    def step(self, action):
        """Execute one step in the environment using discrete action value."""
        return self._step_impl(action)