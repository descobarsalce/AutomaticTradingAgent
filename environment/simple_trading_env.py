import gymnasium as gym
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SimpleTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, transaction_cost=0.0, min_transaction_size=0.001, step_size='1D', max_position_pct=0.95,
                 use_position_profit=True, use_holding_bonus=True, use_trading_penalty=True):
        super().__init__()
        self.use_position_profit = use_position_profit
        self.use_holding_bonus = use_holding_bonus
        self.use_trading_penalty = use_trading_penalty

        # Aggregate data to daily timeframe if higher frequency
        if 'date' in data.columns and step_size == '1D':
            data['date'] = pd.to_datetime(data['date'])
            self.data = data.groupby(data['date'].dt.date).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).reset_index()
        else:
            self.data = data

        # Add tracking for cost basis and last action
        self.cost_basis = 0
        self.last_action = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance  # Initialize cash balance
        self.shares_held = 0            # Initialize position
        self.net_worth = initial_balance  # Initialize net worth
        self.current_step = 0
        self.max_steps = len(data) if data is not None else 100

        # Transaction parameters
        self.transaction_cost = transaction_cost  # As a decimal (e.g., 0.001 for 0.1%)
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

    def _update_net_worth(self) -> float:
        """
        Update and validate the current net worth.
        Returns:
            float: Current net worth
        """
        current_price = self.data.iloc[self.current_step]['Close']
        position_value = self.shares_held * current_price
        new_net_worth = self.balance + position_value

        # Validate net worth
        if new_net_worth < 0:
            logger.warning(f"Negative net worth detected: {new_net_worth}")
            new_net_worth = 0  # Prevent negative net worth
            raise ValueError("Net worth cannot be negative.")

        # Log significant changes (more than 5%)
        if hasattr(self, 'net_worth') and self.net_worth > 0:
            change_pct = (new_net_worth - self.net_worth) / self.net_worth * 100
            if abs(change_pct) > 5:
                logger.info(f"Significant net worth change: {change_pct:.2f}%")

        self.net_worth = new_net_worth
        return self.net_worth

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

        # Reset financial state
        self.balance = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.last_action = 0

        # Calculate initial net worth (should equal initial balance at start)
        self._update_net_worth()

        # Validate initial state
        if self.net_worth != self.initial_balance:
            logger.error(f"Net worth initialization error: {self.net_worth} != {self.initial_balance}")
            raise ValueError("Net worth initialization failed")

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
        # Get current price from market data
        current_price = self.data.iloc[self.current_step]['Close']

        # Ensure action is an integer
        action = int(action)
        if not 0 <= action <= 2:
            raise ValueError(f"Invalid action {action}. Must be 0 (hold), 1 (buy), or 2 (sell)")

        # Fixed position sizing (20% of balance for each trade)
        trade_amount = self.balance * 0.2

        # Process discrete actions
        if action == 1:  # Buy
            shares_to_buy = trade_amount / current_price
            transaction_fees = trade_amount * self.transaction_cost
            total_cost = trade_amount + transaction_fees

            if total_cost <= self.balance:  # Check if we have enough balance
                self.balance -= total_cost
                self.shares_held += shares_to_buy
                self.cost_basis = current_price

        elif action == 2:  # Sell
            if self.shares_held > 0:  # Only sell if we have shares
                shares_to_sell = self.shares_held * 0.2  # Sell 20% of current holdings
                sell_amount = shares_to_sell * current_price
                transaction_fees = sell_amount * self.transaction_cost
                net_sell_amount = sell_amount - transaction_fees

                self.balance += net_sell_amount
                self.shares_held -= shares_to_sell

        # action == 0 is hold, no action needed

        # Store previous net worth for reward calculation
        prev_net_worth = self.net_worth

        # Update portfolio net worth
        self._update_net_worth()

        # Calculate base reward from step-to-step performance
        base_reward = 0
        if prev_net_worth > 0:  # Avoid division by zero
            step_return = (self.net_worth - prev_net_worth) / prev_net_worth
            base_reward = step_return * 100  # Scale up small percentage changes

        # Calculate position profitability
        position_profit = 0
        if self.shares_held > 0 and self.cost_basis > 0:
            profit_pct = (current_price - self.cost_basis) / self.cost_basis
            position_profit = np.clip(profit_pct, -1, 1)

        # Enhanced holding bonus calculation for profitable positions
        holding_bonus = 0
        if self.shares_held > 0:
            # Calculate holding duration with guaranteed positive value
            holding_duration = max(1, self.current_step - self.last_action if self.last_action >= 0 else self.current_step)

            # Calculate position size ratio (0 to 1)
            position_size_ratio = min(1.0, (self.shares_held * current_price) / self.net_worth)

            if position_profit > 0:
                # Monotonically increasing bonus with holding duration
                base_bonus = 0.01 * holding_duration  # 1% per step base bonus
                profit_multiplier = 1 + position_profit  # Scale bonus by profitability
                size_multiplier = 0.5 + (0.5 * position_size_ratio)  # Scale bonus by position size

                # Combine components with guaranteed increase over time
                holding_bonus = base_bonus * profit_multiplier * size_multiplier

        # Trading penalty for frequent trading
        trading_penalty = 0
        if action != 0:  # Apply penalty only for buy/sell actions
            # Size-based penalty
            size_penalty = 0.01  # Fixed penalty for any trade

            # Higher penalty for frequent trading
            frequency_penalty = 0.02 if (self.current_step - self.last_action) < 3 else 0

            trading_penalty = min(size_penalty + frequency_penalty, 0.1)

        # Update last action tracking
        if action != 0:
            self.last_action = self.current_step

        # Combine reward components based on configuration
        reward = base_reward

        if self.use_position_profit:
            reward += position_profit * 0.3  # 30% weight to position profitability

        if self.use_holding_bonus:
            reward += holding_bonus  # Add the monotonically increasing holding bonus

        if self.use_trading_penalty and trading_penalty > 0:
            reward *= (1 - trading_penalty)  # Apply penalty as a reduction factor

        # Update state
        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False

        next_state = self._get_observation()

        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': current_price,
            'trade_status': 'execute',
            'holding_bonus': holding_bonus,  # Add for debugging
            'position_profit': position_profit,
            'trading_penalty': trading_penalty
        }

        return next_state, reward, terminated, truncated, info