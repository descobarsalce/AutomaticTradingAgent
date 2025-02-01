"""
TradingEnv implements a gymnasium-compatible environment for reinforcement learning in trading.
Features efficient vectorized operations and comprehensive error handling.
"""
import gymnasium as gym
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from gymnasium import spaces
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TradingEnv(gym.Env):
    """
    Consolidated trading environment that combines features from all implementations:
    - Multi-asset support 
    - Advanced reward shaping
    - Comprehensive logging and tracking
    - Efficient vectorized operations
    """

    def __init__(self,
                 data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 initial_balance: float = 10000,
                 transaction_cost: float = 0.0,
                 position_size: float = 0.2,
                 use_position_profit: bool = False,
                 use_holding_bonus: bool = False,
                 use_trading_penalty: bool = False,
                 training_mode: bool = False,
                 log_frequency: int = 30,
                 stock_names: Optional[List[str]] = None):
        super().__init__()

        # Validate and process input data
        try:
            self.data = self._process_input_data(data, stock_names)
            self.symbols = list(self.data.keys())

            # Initialize trading parameters
            self.initial_balance = float(initial_balance)
            self.transaction_cost = float(transaction_cost)
            self.position_size = float(position_size)
            self.training_mode = training_mode
            self.log_frequency = int(log_frequency)

            # Feature flags
            self.use_position_profit = use_position_profit
            self.use_holding_bonus = use_holding_bonus
            self.use_trading_penalty = use_trading_penalty

            # Set up action and observation spaces
            self._setup_spaces()

            # Initialize state variables
            self.reset()

        except Exception as e:
            logger.error(f"Environment initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize trading environment: {str(e)}")

    def _process_input_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                          stock_names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Process and validate input data."""
        try:
            if isinstance(data, dict):
                return data

            # If single DataFrame is passed
            if isinstance(data, pd.DataFrame):
                if hasattr(data, 'columns') and len(data.columns.get_level_values(0).unique()) > 0:
                    asset_name = data.columns.get_level_values(0).unique()[0]
                else:
                    asset_name = stock_names[0] if stock_names else "ASSET"
                return {asset_name: data}

            raise ValueError(f"Invalid data type: {type(data)}")

        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            raise

    def _setup_spaces(self):
        """Set up action and observation spaces."""
        try:
            # Action space: 0=hold, 1=buy, 2=sell for each asset
            self.action_space = spaces.MultiDiscrete([3] * len(self.symbols)) if len(self.symbols) > 1 else spaces.Discrete(3)

            # Observation space: OHLCV + positions + balance for each asset
            obs_dim = (len(self.symbols) * 6) + 1  # OHLCV + position for each asset + balance
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        except Exception as e:
            logger.error(f"Failed to setup spaces: {str(e)}")
            raise

    def _get_observation(self) -> np.ndarray:
        """Get current observation of market and account state."""
        try:
            obs = []
            for symbol in self.symbols:
                data = self.data[symbol].iloc[self.current_step]
                obs.extend([
                    float(data['Open']),
                    float(data['High']),
                    float(data['Low']),
                    float(data['Close']),
                    float(data['Volume']),
                    float(self.positions[symbol])
                ])
            obs.append(float(self.balance))
            return np.array(obs, dtype=np.float32)

        except Exception as e:
            logger.error(f"Failed to get observation: {str(e)}")
            raise

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment state."""
        super().reset(seed=seed)

        # Reset portfolio tracking
        self._portfolio_history = []
        self._trade_history = []

        # Reset trading state
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.cost_bases = {symbol: 0.0 for symbol in self.symbols}
        self.holding_periods = {symbol: 0 for symbol in self.symbols}
        self.net_worth = self.initial_balance
        self.episode_trades = {symbol: 0 for symbol in self.symbols}

        # Get initial observation
        observation = self._get_observation()

        info = {
            'initial_balance': self.initial_balance,
            'net_worth': self.net_worth,
            'positions': self.positions.copy(),
            'balance': self.balance
        }

        return observation, info

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment using vectorized operations where possible."""
        try:
            # Store previous state
            prev_net_worth = self.net_worth
            trades_executed = {symbol: False for symbol in self.symbols}

            # Convert action to list if single integer
            actions = [action] if isinstance(action, (int, np.integer)) else action
            if len(actions) != len(self.symbols):
                raise ValueError(f"Expected {len(self.symbols)} actions, got {len(actions)}")

            # Process actions for each asset (vectorized where possible)
            for idx, symbol in enumerate(self.symbols):
                current_price = float(self.data[symbol].iloc[self.current_step]['Close'])
                action = int(actions[idx])

                if action == 1:  # Buy
                    self._execute_buy(symbol, current_price, trades_executed)
                elif action == 2:  # Sell
                    self._execute_sell(symbol, current_price, trades_executed)

                # Update holding period
                if self.positions[symbol] > 0:
                    self.holding_periods[symbol] += 1

            # Update portfolio value (vectorized)
            self.net_worth = self.balance + sum(
                self.positions[symbol] * self.data[symbol].iloc[self.current_step]['Close']
                for symbol in self.symbols
            )
            self._portfolio_history.append(self.net_worth)

            # Calculate reward
            reward = self._compute_reward(prev_net_worth, self.net_worth, actions, trades_executed)

            # Update state
            self.current_step += 1
            done = self.current_step >= len(next(iter(self.data.values()))) - 1

            # Get next observation
            observation = self._get_observation()

            # Prepare info dict
            info = {
                'net_worth': self.net_worth,
                'balance': self.balance,
                'positions': self.positions.copy(),
                'trades_executed': trades_executed,
                'date': self.data[self.symbols[0]].iloc[self.current_step].name
            }

            # Store trade history
            self._trade_history.append(info)

            return observation, reward, done, False, info

        except Exception as e:
            logger.error(f"Error in step execution: {str(e)}")
            raise

    def _execute_buy(self, symbol: str, current_price: float, trades_executed: Dict[str, bool]):
        """Execute a buy order."""
        try:
            max_trade_amount = max(0, self.balance - self.transaction_cost)
            trade_amount = min(max_trade_amount * self.position_size, max_trade_amount)
            shares_to_buy = trade_amount / current_price if current_price > 0 else 0
            total_cost = (shares_to_buy * current_price) + self.transaction_cost

            if total_cost <= self.balance and shares_to_buy >= 0.01:
                self.balance -= total_cost
                if self.positions[symbol] > 0:
                    # Update cost basis
                    old_cost = self.cost_bases[symbol] * self.positions[symbol]
                    new_cost = current_price * shares_to_buy
                    self.cost_bases[symbol] = (old_cost + new_cost) / (self.positions[symbol] + shares_to_buy)
                else:
                    self.cost_bases[symbol] = current_price

                self.positions[symbol] += shares_to_buy
                self.holding_periods[symbol] = 0
                self.episode_trades[symbol] += 1
                trades_executed[symbol] = True

                if self.training_mode:
                    logger.debug(f"BUY | {symbol} | Price: ${current_price:.2f} | Shares: {shares_to_buy:.4f}")

        except Exception as e:
            logger.error(f"Buy execution failed: {str(e)}")
            raise

    def _execute_sell(self, symbol: str, current_price: float, trades_executed: Dict[str, bool]):
        """Execute a sell order."""
        try:
            if self.positions[symbol] > 0:
                shares_to_sell = self.positions[symbol]
                sell_amount = shares_to_sell * current_price
                net_sell_amount = sell_amount - self.transaction_cost

                self.balance += net_sell_amount
                self.positions[symbol] = 0
                self.holding_periods[symbol] = 0
                self.episode_trades[symbol] += 1
                trades_executed[symbol] = True

                if self.training_mode:
                    logger.debug(f"SELL | {symbol} | Price: ${current_price:.2f} | Amount: ${net_sell_amount:.2f}")

        except Exception as e:
            logger.error(f"Sell execution failed: {str(e)}")
            raise

    def _compute_reward(self, prev_net_worth: float, current_net_worth: float,
                       actions: Union[int, np.ndarray], trades_executed: Dict[str, bool]) -> float:
        """Calculate reward based on trading performance and behavior."""
        try:
            reward = 0.0

            # Portfolio return component
            if prev_net_worth > 0:
                portfolio_return = (current_net_worth - prev_net_worth) / prev_net_worth
                reward += portfolio_return

            # Additional reward components based on flags
            if self.use_holding_bonus:
                # Reward for holding profitable positions
                for symbol in self.symbols:
                    if self.positions[symbol] > 0:
                        current_price = float(self.data[symbol].iloc[self.current_step]['Close'])
                        if current_price > self.cost_bases[symbol]:
                            reward += 0.001 * self.holding_periods[symbol]

            if self.use_trading_penalty:
                # Penalize excessive trading
                trade_penalty = sum(trades_executed.values()) * 0.0001
                reward -= trade_penalty

            return float(reward)

        except Exception as e:
            logger.error(f"Reward computation failed: {str(e)}")
            raise

    def get_portfolio_history(self) -> List[float]:
        """Return the history of portfolio values."""
        return self._portfolio_history

    def get_trade_history(self) -> List[Dict]:
        """Return the history of trades."""
        return self._trade_history