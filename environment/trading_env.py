import gymnasium as gym
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from gymnasium import spaces
from utils.common import (MAX_POSITION_SIZE, MIN_POSITION_SIZE, MIN_TRADE_SIZE,
                         POSITION_PRECISION)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TradingEnv(gym.Env):
    """A consolidated trading environment supporting multiple assets with enhanced reward shaping."""

    def __init__(self,
                 data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 initial_balance: float = 10000,
                 transaction_cost: float = 0.0,
                 position_size: float = 0.2,
                 use_position_profit: bool = False,
                 use_holding_bonus: bool = False,
                 use_trading_penalty: bool = False,
                 training_mode: bool = False):
        """Initialize the trading environment with validation."""
        super().__init__()
        self._validate_init_params(data, initial_balance, transaction_cost, position_size)

        self._full_data = data.copy()
        self.data = data
        self.symbols = self._extract_symbols()
        self.position_size = position_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        # Reward shaping flags
        self.use_position_profit = use_position_profit
        self.use_holding_bonus = use_holding_bonus
        self.use_trading_penalty = use_trading_penalty
        self.training_mode = training_mode

        # Action and observation spaces
        self.action_space = self._create_action_space()
        self.observation_space = self._create_observation_space()

        # Initialize trading state
        self._initialize_state()

    def _validate_init_params(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                            initial_balance: float, transaction_cost: float,
                            position_size: float) -> None:
        """Validate initialization parameters."""
        if initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        if transaction_cost < 0:
            raise ValueError("Transaction cost cannot be negative")
        if not 0 < position_size <= 1:
            raise ValueError("Position size must be between 0 and 1")
        if isinstance(data, pd.DataFrame) and data.empty:
            raise ValueError("Empty data provided")

    def _extract_symbols(self) -> List[str]:
        """Extract trading symbols from data columns."""
        try:
            symbols = sorted(list({col.split('_')[1] for col in self.data.columns 
                                 if '_' in col and len(col.split('_')) == 2}))
            if not symbols:
                raise ValueError("No valid symbols found in data columns")
            return symbols
        except Exception as e:
            logger.error(f"Error extracting symbols from columns: {self.data.columns}")
            raise ValueError(f"Failed to extract valid symbols from data columns: {e}")

    def _create_action_space(self) -> gym.Space:
        """Create action space for trading environment."""
        return spaces.MultiDiscrete([3] * len(self.symbols)) if len(self.symbols) > 1 else spaces.Discrete(3)

    def _create_observation_space(self) -> spaces.Box:
        """Create observation space for market state."""
        obs_dim = (len(self.symbols) * 2) + 1  # prices + positions + balance
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _initialize_state(self) -> None:
        """Initialize or reset trading state."""
        self._portfolio_history = []
        self._trade_history = []
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.cost_bases = {symbol: 0.0 for symbol in self.symbols}
        self.holding_periods = {symbol: 0 for symbol in self.symbols}
        self.net_worth = self.initial_balance
        self.episode_trades = {symbol: 0 for symbol in self.symbols}

    def _get_observation(self) -> np.ndarray:
        """Get current market observation."""
        obs = []
        for symbol in self.symbols:
            current_price = float(self._full_data.iloc[self.current_step][f'Close_{symbol}'])
            obs.extend([current_price, self.positions[symbol]])
        obs.append(self.balance)
        return np.array(obs, dtype=np.float32)

    def _execute_trades(self, actions: Union[int, np.ndarray]) -> Dict[str, bool]:
        """Execute trading actions and return execution status."""
        trades_executed = {symbol: False for symbol in self.symbols}
        actions_list = [actions] if isinstance(actions, (int, np.integer)) else actions

        for idx, (symbol, action) in enumerate(zip(self.symbols, actions_list)):
            current_price = float(self.data.iloc[self.current_step][f'Close_{symbol}'])

            if action == 1:  # Buy
                trades_executed[symbol] = self._execute_buy(symbol, current_price)
            elif action == 2:  # Sell
                trades_executed[symbol] = self._execute_sell(symbol, current_price)

            if self.positions[symbol] > 0:
                self.holding_periods[symbol] += 1

        return trades_executed

    def _execute_buy(self, symbol: str, current_price: float) -> bool:
        """Execute buy order for given symbol."""
        max_trade_amount = max(0, self.balance - self.transaction_cost)
        trade_amount = min(max_trade_amount * self.position_size, self.balance)

        if trade_amount < MIN_TRADE_SIZE:
            return False

        shares_to_buy = trade_amount / current_price if current_price > 0 else 0
        total_cost = (shares_to_buy * current_price) + self.transaction_cost

        if total_cost <= self.balance and shares_to_buy >= 0.01:
            self.balance -= total_cost
            if self.positions[symbol] > 0:
                # Update cost basis for additional purchases
                old_cost = self.cost_bases[symbol] * self.positions[symbol]
                new_cost = current_price * shares_to_buy
                self.cost_bases[symbol] = (old_cost + new_cost) / (self.positions[symbol] + shares_to_buy)
            else:
                self.cost_bases[symbol] = current_price
            self.positions[symbol] = round(self.positions[symbol] + shares_to_buy, POSITION_PRECISION)
            self.episode_trades[symbol] += 1
            return True
        return False

    def _execute_sell(self, symbol: str, current_price: float) -> bool:
        """Execute sell order for given symbol."""
        if self.positions[symbol] <= 0:
            return False

        shares_to_sell = (self.positions[symbol]/2 if self.positions[symbol]*current_price >= MIN_TRADE_SIZE 
                         else self.positions[symbol])
        sell_amount = shares_to_sell * current_price
        net_sell_amount = sell_amount - self.transaction_cost

        if net_sell_amount > 0:
            self.balance += net_sell_amount
            self.positions[symbol] = round(self.positions[symbol] - shares_to_sell, POSITION_PRECISION)
            self.episode_trades[symbol] += 1
            return True
        return False

    def _compute_reward(self, prev_net_worth: float, current_net_worth: float,
                       actions: Union[int, np.ndarray], trades_executed: Dict[str, bool]) -> float:
        """Calculate trading reward with optional shaping."""
        reward = 0.0

        # Base reward from portfolio return
        if prev_net_worth > 0:
            portfolio_return = (current_net_worth - prev_net_worth) / prev_net_worth
            reward += portfolio_return

        # Additional reward components based on flags
        if self.use_holding_bonus:
            reward += self._calculate_holding_bonus()
        if self.use_trading_penalty:
            reward -= sum(trades_executed.values()) * 0.0001
        if not any(trades_executed.values()):
            reward -= 0.01  # Small penalty for no trades

        return reward

    def _calculate_holding_bonus(self) -> float:
        """Calculate bonus for holding profitable positions."""
        bonus = 0.0
        for symbol in self.symbols:
            current_price = float(self.data.iloc[self.current_step][f'Close_{symbol}'])
            if self.positions[symbol] > 0 and current_price > self.cost_bases[symbol]:
                bonus += 0.001 * self.holding_periods[symbol]
        return bonus

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        prev_net_worth = self.net_worth
        trades_executed = self._execute_trades(action)

        # Update portfolio value
        self.net_worth = self.balance + sum(
            self.positions[symbol] * float(self.data.iloc[self.current_step][f'Close_{symbol}'])
            for symbol in self.symbols)
        self._portfolio_history.append(self.net_worth)

        # Calculate reward
        reward = self._compute_reward(prev_net_worth, self.net_worth, action, trades_executed)

        # Update state
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # Prepare info dict
        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'trades_executed': trades_executed,
            'episode_trades': self.episode_trades.copy(),
            'actions': action,
            'date': self.data.index[self.current_step],
            'current_data': self.data.iloc[self.current_step]
        }
        self._trade_history.append(info)

        return self._get_observation(), reward, done, False, info

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        self._initialize_state()

        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'trades_executed': {symbol: False for symbol in self.symbols},
            'episode_trades': self.episode_trades,
            'actions': {},
            'date': self.data.index[self.current_step],
            'current_data': self.data.iloc[self.current_step]
        }

        return self._get_observation(), info

    def get_portfolio_history(self) -> List[float]:
        """Return the history of portfolio values."""
        return self._portfolio_history