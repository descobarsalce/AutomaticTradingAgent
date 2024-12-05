import gym
import numpy as np
from gym import spaces
import pandas as pd

class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=100000):
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # Action space: continuous value between -1 (full sell) and 1 (full buy)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation space: price data + account info
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(7,), # OHLCV + position + balance
            dtype=np.float32
        )
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.cost_basis = 0
        return self._get_observation()
        
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
        
    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        
        # Execute trade
        action = action[0]
        amount = self.balance * abs(action)
        if action > 0:  # Buy
            shares_bought = amount / current_price
            self.balance -= amount
            self.shares_held += shares_bought
            self.cost_basis = current_price
        elif action < 0:  # Sell
            shares_sold = self.shares_held * abs(action)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            
        # Calculate reward
        self.net_worth = self.balance + self.shares_held * current_price
        reward = (self.net_worth - self.initial_balance) / self.initial_balance
        
        # Update state
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        obs = self._get_observation()
        
        return obs, reward, done, {}
