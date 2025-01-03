import gymnasium as gym
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SimpleTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, transaction_cost=0.0, min_transaction_size=0.001, 
                 max_position_pct=0.95, use_position_profit=True, use_holding_bonus=True, use_trading_penalty=True,
                 training_mode=False):
        super().__init__()
        self.use_position_profit = use_position_profit
        self.use_holding_bonus = use_holding_bonus
        self.use_trading_penalty = use_trading_penalty
        self.training_mode = training_mode  # Flag to enable training-specific behavior

        # Store data
        self.data = data

        # Initialize trading state
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.current_step = 0
        self.last_logged_step = -1
        self.max_steps = len(data) if data is not None else 100

        # Track holding period and cost basis
        self.holding_period = 0
        self.cost_basis = 0
        self.last_trade_step = None
        self.episode_trades = 0

        # Transaction parameters
        self.base_transaction_cost = transaction_cost
        self.transaction_cost = transaction_cost * (0.2 if training_mode else 1.0)  # Reduced cost during training
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

        # Episode tracking
        self.episode_count = 0
        self.total_steps = 0

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
        self.episode_trades = 0
        self.episode_count += 1

        observation = self._get_observation()
        info = {
            'initial_balance': self.initial_balance,
            'net_worth': self.net_worth,
            'shares_held': self.shares_held,
            'balance': self.balance,
            'episode': self.episode_count,
            'total_steps': self.total_steps
        }
        return observation, info

    def step(self, action):
        """Execute one step in the environment."""
        self.total_steps += 1

        # Log action for debugging
        logger.debug(f"Step {self.current_step}, Action: {action}, Balance: {self.balance:.2f}, Shares: {self.shares_held:.2f}")

        # Ensure action is valid
        action = int(action)
        if not 0 <= action <= 2:
            raise ValueError(f"Invalid action {action}. Must be 0 (hold), 1 (buy), or 2 (sell)")

        # Store previous state and price
        prev_net_worth = self.net_worth
        current_price = float(self.data.iloc[self.current_step]['Close'])

        # Process actions
        trade_executed = False
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
                self.episode_trades += 1
                trade_executed = True
                logger.info(f"""
Trade Executed - BUY:
  Shares: {shares_to_buy:.4f}
  Price: {current_price:.2f}
  Amount: {trade_amount:.2f}
  Fees: {transaction_fees:.2f}
  Total Cost: {total_cost:.2f}
""")

        elif action == 2:  # Sell
            if self.shares_held > 0:
                shares_to_sell = self.shares_held * 0.2  # Sell 20% of holdings
                sell_amount = shares_to_sell * current_price
                transaction_fees = sell_amount * self.transaction_cost
                net_sell_amount = sell_amount - transaction_fees

                self.balance += net_sell_amount
                self.shares_held -= shares_to_sell
                self.episode_trades += 1
                trade_executed = True
                logger.info(f"""
Trade Executed - SELL:
  Shares: {shares_to_sell:.4f}
  Price: {current_price:.2f}
  Amount: {sell_amount:.2f}
  Fees: {transaction_fees:.2f}
  Net Amount: {net_sell_amount:.2f}
""")
                if self.shares_held == 0:
                    self.holding_period = 0
                    self.last_trade_step = None

        # Update portfolio value
        self.net_worth = self.balance + (self.shares_held * current_price)
        
        # Log portfolio state only when trade is executed and haven't logged this step yet
        if trade_executed and self.current_step > self.last_logged_step:
            self.last_logged_step = self.current_step
            logger.info(f"""
Portfolio State:
  Step: {self.current_step}
  Action: {action}
  Price: {current_price:.2f}
  Balance: {self.balance:.2f}
  Shares: {self.shares_held:.4f}
  Net Worth: {self.net_worth:.2f}
  Change: {((self.net_worth - prev_net_worth) / prev_net_worth * 100):.2f}% 
  Position Value: {(self.shares_held * current_price):.2f}
""")

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
        base_reward = 0
        position_profit_reward = 0
        holding_bonus = 0

        # 1. Base reward from portfolio change
        if prev_net_worth > 0:
            portfolio_return = (self.net_worth - prev_net_worth) / prev_net_worth
            base_reward = portfolio_return
            reward += base_reward

        # 2. Position profit component
        if self.use_position_profit and self.shares_held > 0:
            position_profit_reward = position_profit  # Already calculated above
            reward += position_profit_reward

        # 3. Holding bonus
        if self.use_holding_bonus and self.shares_held > 0:
            holding_bonus = 0.001 * self.holding_period * position_profit if position_profit > 0 else 0
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

        # 5. Early exploration bonus during training
        if self.training_mode and self.episode_count < 10 and trade_executed:
            exploration_bonus = 0.1  # Small bonus for executing trades early in training
            reward += exploration_bonus

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
            'episode_trades': self.episode_trades,
            'action_taken': action,
            'trade_executed': trade_executed,
            'reward_components': {
                'base': base_reward if 'base_reward' in locals() else 0,
                'position_profit': position_profit * 0.05 if self.use_position_profit else 0,
                'holding_bonus': holding_bonus if 'holding_bonus' in locals() else 0,
                'trading_penalty': 'applied' if self.use_trading_penalty and action != 0 else 'none',
                'exploration_bonus': exploration_bonus if 'exploration_bonus' in locals() else 0
            }
        }

        # Log reward components
        logger.info(f"""
Reward Components:
  Base Reward: {base_reward if 'base_reward' in locals() else 0:.4f}
  Position Profit: {position_profit * 0.01 if self.use_position_profit else 0:.4f}
  Holding Bonus: {holding_bonus if 'holding_bonus' in locals() else 0:.4f}
  Final Reward: {reward:.4f}
""")

        return next_observation, reward, done, truncated, info

    def test_random_actions(self, n_steps: int = 100) -> Dict[str, Union[int, float]]:
        """
        Test environment with random actions to verify functionality.

        Args:
            n_steps: Number of random steps to take

        Returns:
            Dict containing test statistics
        """
        obs, _ = self.reset()
        total_trades = 0
        total_rewards = 0
        actions_taken = {0: 0, 1: 0, 2: 0}  # Count of each action type
        trades_executed = 0

        for _ in range(n_steps):
            action = self.action_space.sample()
            obs, reward, done, truncated, info = self.step(action)

            actions_taken[action] += 1
            if info['trade_executed']:
                trades_executed += 1
            total_rewards += reward

            if done:
                obs, _ = self.reset()

        return {
            'total_steps': n_steps,
            'actions_taken': actions_taken,
            'trades_executed': trades_executed,
            'avg_reward': total_rewards / n_steps,
            'trade_success_rate': trades_executed / sum(actions_taken[i] for i in [1, 2]) if sum(actions_taken[i] for i in [1, 2]) > 0 else 0
        }