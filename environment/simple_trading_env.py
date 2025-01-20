import gymnasium as gym
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SimpleTradingEnv(gym.Env):

    def __init__(self,
                 data,
                 initial_balance=10000,
                 transaction_cost=0.0,
                 use_position_profit=False,
                 use_holding_bonus=False,
                 use_trading_penalty=False,
                 training_mode=False,
                 log_frequency=30):
        super().__init__()
        self._portfolio_history = []  # Track portfolio value over time
        self.use_position_profit = use_position_profit
        self.use_holding_bonus = use_holding_bonus
        self.use_trading_penalty = use_trading_penalty
        self.training_mode = training_mode  # Flag to enable training-specific behavior
        self.log_frequency = log_frequency  # How often to log portfolio state

        # Store data
        self.data = data

        # Define action space as discrete: 0 (hold), 1 (buy), 2 (sell)
        self.action_space = gym.spaces.Discrete(3)

        # Define observation space for OHLCV + position + balance
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7, ),  # OHLCV + position + balance
            dtype=np.float32)

        # Initialize trading state
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.current_step = 0
        self.last_logged_step = -1
        self.max_steps = len(data) if data is not None else 100
        self.episode_count = 0
        self.total_steps = 0

        # Track holding period and cost basis
        self.holding_period = 0
        self.cost_basis = 0
        self.last_trade_step = None
        self.episode_trades = 0

        # Transaction parameters
        self.transaction_cost = transaction_cost

    def _compute_reward(self, prev_net_worth: float, current_net_worth: float,
                        action: int, trade_executed: bool) -> float:
        """
        Calculate the reward based on trading performance and behavior.
        
        Args:
            prev_net_worth: Previous portfolio value
            current_net_worth: Current portfolio value 
            action: Trading action taken (0=hold, 1=buy, 2=sell)
            position_profit: Profit/loss on current position
            holding_period: Number of steps current position is held
            trade_executed: Whether a trade was executed
            
        Returns:
            float: Calculated reward value
        """
        reward = 0
        base_reward = 0

        # Base reward from portfolio return
        if prev_net_worth > 0:
            portfolio_return = (current_net_worth -
                                prev_net_worth) / prev_net_worth
            base_reward = portfolio_return
            reward += base_reward
        else:
            logger.warning(
                "Previous portfolio value is zero. Skipping reward calculation."
            )

        return reward

    def _get_observation(self):
        """Get current observation of market and account state."""
        obs = np.array([
            self.data.iloc[self.current_step]['Open'],
            self.data.iloc[self.current_step]['High'],
            self.data.iloc[self.current_step]['Low'],
            self.data.iloc[self.current_step]['Close'],
            self.data.iloc[self.current_step]['Volume'], self.shares_held,
            self.balance
        ],
                       dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self._portfolio_history = []  # Reset portfolio history
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        # self.holding_period = 0
        self.last_logged_step = -1  # Reset logging tracker
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

    @property
    def _initial_balance(self):
        """Property to maintain backward compatibility"""
        return self.initial_balance

    def step(self, action):
        """Execute one step in the environment."""
        self.total_steps += 1

        # Log action for debugging
        logger.debug(
            f"Step {self.current_step}, Action: {action}, Balance: {self.balance:.2f}, Shares: {self.shares_held:.2f}"
        )

        # Ensure action is valid
        action = int(action)
        if not 0 <= action <= 2:
            raise ValueError(
                f"Invalid action {action}. Must be 0 (hold), 1 (buy), or 2 (sell)"
            )

        # Store previous state and price
        prev_net_worth = self.net_worth
        current_price = float(self.data.iloc[self.current_step]['Close'])

        # Process actions
        trade_executed = False
        if action == 1:  # Buy
            trade_amount = self.balance - self.transaction_cost
            shares_to_buy = trade_amount / current_price
            transaction_fees = self.transaction_cost
            total_cost = trade_amount + transaction_fees

            if total_cost <= self.balance:
                self.balance -= total_cost
                if self.shares_held > 0:
                    # Calculate weighted average cost basis
                    old_cost = self.cost_basis * self.shares_held
                    new_cost = current_price * shares_to_buy
                    self.cost_basis = (old_cost + new_cost) / (
                        self.shares_held + shares_to_buy)
                else:
                    # Initial position cost basis
                    self.cost_basis = current_price
                self.shares_held += shares_to_buy
                # self.holding_period = 0
                self.last_trade_step = self.current_step
                self.episode_trades += 1
                trade_executed = True

        elif action == 2:  # Sell
            if self.shares_held > 0:
                shares_to_sell = self.shares_held
                sell_amount = shares_to_sell * current_price
                transaction_fees = self.transaction_cost
                net_sell_amount = sell_amount

                self.balance += net_sell_amount
                self.shares_held -= shares_to_sell
                self.episode_trades += 1
                trade_executed = True

                if self.shares_held == 0:
                    # self.holding_period = 0
                    self.last_trade_step = None

        # Update portfolio value
        self.net_worth = self.balance + (self.shares_held * current_price)
        self._portfolio_history.append(self.net_worth)

        # Decide whether to log portfolio state (single consolidated check)
        if (self.current_step - self.last_logged_step) >= self.log_frequency:
            self.last_logged_step = self.current_step

        # Update holding period for any non-zero position
        if self.shares_held > 0:
            self.holding_period += 1

        # Calculate reward using dedicated method
        reward = self._compute_reward(prev_net_worth=prev_net_worth,
                                      current_net_worth=self.net_worth,
                                      action=action,
                                      trade_executed=trade_executed)

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
            'episode_trades': self.episode_trades,
            'action_taken': action,
            'trade_executed': trade_executed,
            'reward_components': {
                'base': base_reward if 'base_reward' in locals() else 0
            }
        }

        return next_observation, reward, done, truncated, info

    def get_portfolio_history(self):
        """Return the history of portfolio values"""
        return self._portfolio_history

    def test_random_actions(self,
                            n_steps: int = 512,
                            log: bool = False) -> Dict[str, Union[int, float]]:
        """
        Test environment with random actions to verify functionality.

        Args:
            n_steps: Number of random steps to take
            log: Whether to allow logging during testing

        Returns:
            Dict containing test statistics
        """
        # Store original logging level
        original_level = logger.getEffectiveLevel()

        try:
            # Set logging level based on log parameter
            if not log:
                logger.setLevel(logging.WARNING)

            obs, _ = self.reset()
            total_rewards = 0
            actions_taken = {0: 0, 1: 0, 2: 0}  # Count of each action type
            trades_executed = 0

            for step in range(n_steps):
                action = self.action_space.sample()
                obs, reward, done, truncated, info = self.step(action)

                actions_taken[action] += 1
                if info['trade_executed']:
                    trades_executed += 1
                total_rewards += reward

                if done:
                    obs, _ = self.reset()

            results = {
                'total_steps':
                n_steps,
                'actions_taken':
                actions_taken,
                'trades_executed':
                trades_executed,
                'avg_reward':
                total_rewards / n_steps,
                'trade_success_rate':
                trades_executed / sum(actions_taken[i] for i in [1, 2]) if sum(
                    actions_taken[i] for i in [1, 2]) > 0 else 0
            }

            if log:
                logger.info(f"Test results: {results}")

            return results

        finally:
            # Restore original logging level
            logger.setLevel(original_level)
