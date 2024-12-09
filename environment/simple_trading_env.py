import gymnasium as gym
import numpy as np
import gymnasium as gym
import pandas as pd
from typing import Tuple, Dict, Any, Optional

import numpy as np

class SimpleTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=100000):
        super().__init__()
        # Store market data and initial balance
        self.data = data
        # Add tracking for cost basis and last action
        self.cost_basis = 0
        self.last_action = 0
        self.initial_balance = initial_balance
        self.current_step = 0
        self.max_steps = len(data) if data is not None else 100
        
        # Define action space as continuous values between -1 and 1
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
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
        try:
            super().reset(seed=seed)
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
            
        except Exception as e:
            print(f"Error in reset method: {str(e)}")
            raise
        
    def step(self, action):
        """Execute one step in the environment."""
        try:
            # Get current price from market data
            current_price = self.data.iloc[self.current_step]['Close']
            
            # Validate and process action
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            action = float(action[0])  # Extract single action value
            action = np.clip(action, -1.0, 1.0)  # Ensure action is in [-1, 1]
            
            # Calculate position sizing based on continuous action
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
            
            # Calculate base reward from portfolio performance
            self.net_worth = self.balance + self.shares_held * current_price
            base_reward = (self.net_worth - self.initial_balance) / self.initial_balance
            
            # Calculate position profitability with validation
            position_profit = 0
            if self.shares_held > 0 and self.cost_basis > 0:
                position_profit = np.clip((current_price - self.cost_basis) / self.cost_basis, -1, 1)
            
            # Calculate market trend using exponential moving average for smoother signals
            trend_window = min(5, self.current_step + 1)
            if trend_window > 1:
                prices = self.data['Close'].iloc[max(0, self.current_step-trend_window+1):self.current_step+1]
                ema = prices.ewm(span=trend_window).mean().iloc[-1]
                trend = np.clip((current_price - ema) / ema, -0.1, 0.1)
            else:
                trend = 0
                
            # Simplified holding bonus with guaranteed monotonic growth
            holding_bonus = 0
            if self.shares_held > 0:
                # Calculate holding duration (always positive)
                holding_duration = max(1, self.current_step - self.last_action if self.last_action >= 0 else self.current_step)
                
                # Calculate position size ratio (0 to 1)
                position_size_ratio = (self.shares_held * current_price) / self.net_worth
                
                if position_profit > 0:
                    # Linear growth with duration
                    duration_bonus = 0.005 * holding_duration  # 0.5% per step
                    # Scale by position profit and size
                    holding_bonus = duration_bonus * (1 + position_profit) * position_size_ratio
            
            # Calculate market volatility using rolling window
            volatility = 0.0
            if self.current_step >= 5:  # Need at least 5 data points
                prices = self.data['Close'].iloc[max(0, self.current_step-20):self.current_step+1]
                returns = np.log(prices / prices.shift(1)).dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Adaptive trading penalty based on action magnitude, frequency, and volatility
            trading_penalty = 0
            if abs(action) > 0.1:
                # Base size penalty scaled by volatility
                volatility_scalar = max(1.0, volatility * 10)  # Higher volatility = higher penalty
                size_penalty = 0.001 * (abs(action) - 0.1) * volatility_scalar
                
                # Frequency penalty increases in high volatility
                min_hold_period = max(3, int(5 * volatility_scalar))  # Dynamic holding period
                frequency_penalty = 0.001 * volatility_scalar if (self.current_step - self.last_action) < min_hold_period else 0
                
                trading_penalty = size_penalty + frequency_penalty
            
            # Update last action tracking for any significant trade
            if abs(action) > 0.1:
                self.last_action = self.current_step
            
            # Combine reward components with emphasis on holding
            reward = (
                base_reward * 0.15 +          # Base portfolio performance (15%)
                position_profit * 0.15 +      # Position profitability (15%)
                trend * 0.10 +                # Market trend alignment (10%)
                holding_bonus * 0.60 -        # Holding bonus (60%)
                trading_penalty               # Trading penalty (reduces reward)
            )
            
            # Update state
            self.current_step += 1
            terminated = self.current_step >= len(self.data) - 1
            truncated = False
            
            next_state = self._get_observation()
            
            info = {
                'net_worth': self.net_worth,
                'balance': self.balance,
                'shares_held': self.shares_held,
                'current_price': current_price
            }
            
            return next_state, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"Error in step method: {str(e)}")
            raise
