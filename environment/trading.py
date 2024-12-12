
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from gymnasium import spaces
from .base import BaseEnvironment

class SimpleTradingEnv(BaseEnvironment):
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0,
                 transaction_cost: float = 0.001, min_transaction_size: float = 10.0):
        super().__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.min_transaction_size = min_transaction_size
        
        # Optimize spaces definition
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.asset_value = 0
        self.total_transaction_costs = 0
        self.previous_net_worth = self.initial_balance
        
        return self._get_observation(), self._get_info()
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Validate current step
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Get current price and execute trade
        current_price = self.data.iloc[self.current_step]['Close']
        trade_shares = self._calculate_trade_shares(action[0], current_price)
        
        # Update position and calculate costs
        transaction_cost = abs(trade_shares * current_price * self.transaction_cost)
        self.total_transaction_costs += transaction_cost
        self.balance -= (trade_shares * current_price + transaction_cost)
        self.shares_held += trade_shares
        self.asset_value = self.shares_held * current_price
        
        # Move to next step and calculate reward
        self.current_step += 1
        current_net_worth = self.balance + self.asset_value
        reward = (current_net_worth - self.previous_net_worth) / self.previous_net_worth
        self.previous_net_worth = current_net_worth
        
        done = self.current_step >= len(self.data) - 1
        return self._get_observation(), reward, done, False, self._get_info()
        
    def _calculate_trade_shares(self, action: float, price: float) -> float:
        available_amount = min(
            self.balance if action > 0 else self.asset_value,
            abs(action) * self.initial_balance
        )
        trade_shares = (available_amount / price) if price > 0 else 0
        trade_shares *= np.sign(action)
        
        if abs(trade_shares * price) < self.min_transaction_size:
            return 0
        return trade_shares
        
    def _get_observation(self) -> np.ndarray:
        return np.array([
            self.balance / self.initial_balance,
            self.shares_held,
            self.data.iloc[self.current_step]['Close'],
            self.total_transaction_costs / self.initial_balance,
            self.asset_value / self.initial_balance
        ], dtype=np.float32)
        
    def _get_info(self) -> Dict[str, Any]:
        return {
            'net_worth': self.balance + self.asset_value,
            'shares_held': self.shares_held,
            'balance': self.balance,
            'transaction_costs': self.total_transaction_costs
        }
