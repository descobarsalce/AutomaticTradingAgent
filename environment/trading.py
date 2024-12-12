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
        
        # Action space: continuous value between -1 (full sell) and 1 (full buy)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
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
        """Core step logic implementation."""
        current_price = self.data.iloc[self.current_step]['Close']
        
        # Ensure action is in correct format and range
        action = float(action[0])  # Extract single action value
        action = np.clip(action, -1.0, 1.0)  # Ensure action is in [-1, 1]
        
        # Calculate position sizing based on continuous action
        amount = self.balance * abs(action)
        
        if action > 0:  # Buy
            # Scale buy amount based on action magnitude
            shares_bought = amount / current_price
            self.balance -= amount
            self.shares_held += shares_bought
            self.cost_basis = current_price
        elif action < 0:  # Sell
            # Scale sell amount based on action magnitude
            shares_sold = self.shares_held * abs(action)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            
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
        """Execute one step in the environment using continuous action value."""
        return self._step_impl(action)  # Already returns the correct Gymnasium format
