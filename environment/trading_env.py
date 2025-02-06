import gymnasium as gym
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from gymnasium import spaces
from utils.common import (MAX_POSITION_SIZE, MIN_POSITION_SIZE, MIN_TRADE_SIZE,
                         POSITION_PRECISION)
from core.portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TradingEnv(gym.Env):
    def __init__(self,
                 data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 initial_balance: float = 10000,
                 transaction_cost: float = 0.0,
                 position_size: float = 0.2,
                 use_position_profit: bool = False,
                 use_holding_bonus: bool = False,
                 use_trading_penalty: bool = False,
                 training_mode: bool = True):
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

        # Initialize portfolio manager
        self.portfolio_manager = PortfolioManager(initial_balance, transaction_cost)
        self._initialize_state()

    def _validate_init_params(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                            initial_balance: float, transaction_cost: float,
                            position_size: float) -> None:
        if initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        if transaction_cost < 0:
            raise ValueError("Transaction cost cannot be negative")
        if not 0 < position_size <= 1:
            raise ValueError("Position size must be between 0 and 1")
        if isinstance(data, pd.DataFrame) and data.empty:
            raise ValueError("Empty data provided")

    def _extract_symbols(self) -> List[str]:
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
        return spaces.MultiDiscrete([3] * len(self.symbols)) if len(self.symbols) > 1 else spaces.Discrete(3)

    def _create_observation_space(self) -> spaces.Box:
        obs_dim = (len(self.symbols) * 2) + 1  # prices + positions + balance
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _initialize_state(self) -> None:
        self.current_step = 0
        self.episode_trades = {symbol: 0 for symbol in self.symbols}

    def _get_observation(self) -> np.ndarray:
        obs = []
        for symbol in self.symbols:
            current_price = float(self._full_data.iloc[self.current_step][f'Close_{symbol}'])
            obs.extend([current_price, self.portfolio_manager.positions.get(symbol, 0.0)])
        obs.append(self.portfolio_manager.current_balance)
        return np.array(obs, dtype=np.float32)

    def _execute_trades(self, actions: Union[int, np.ndarray]) -> Dict[str, bool]:
        trades_executed = {symbol: False for symbol in self.symbols}
        actions_list = [actions] if isinstance(actions, (int, np.integer)) else actions

        for idx, (symbol, action) in enumerate(zip(self.symbols, actions_list)):
            current_price = float(self.data.iloc[self.current_step][f'Close_{symbol}'])
            timestamp = self.data.index[self.current_step]

            if action > 0:  # 1 for buy, 2 for sell
                quantity = self._calculate_trade_quantity(action, symbol, current_price)
                if quantity > 0:
                    trade_executed = self.portfolio_manager.execute_trade(
                        symbol, action, quantity, current_price, timestamp)
                    trades_executed[symbol] = trade_executed
                    if trade_executed:
                        self.episode_trades[symbol] += 1

        return trades_executed

    def _calculate_trade_quantity(self, action: int, symbol: str, price: float) -> float:
        is_buy = action == 1
        if is_buy:
            max_trade_amount = self.portfolio_manager.current_balance * self.position_size
            return max_trade_amount / price if price > 0 else 0
        else:
            current_position = self.portfolio_manager.positions.get(symbol, 0.0)
            return current_position if current_position > 0 else 0

    def _compute_reward(self, trades_executed: Dict[str, bool]) -> float:
        reward = 0.0
        total_value = self.portfolio_manager.get_total_value()

        if len(self._portfolio_history) > 0:
            prev_value = self._portfolio_history[-1]
            if prev_value > 0:
                reward = (total_value - prev_value) / prev_value

        if self.use_trading_penalty:
            reward -= sum(trades_executed.values()) * 0.0001
        if not any(trades_executed.values()):
            reward -= 0.01

        return reward

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        trades_executed = self._execute_trades(action)
        current_value = self.portfolio_manager.get_total_value()
        self._portfolio_history.append(current_value)

        reward = self._compute_reward(trades_executed)
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        info = {
            'net_worth': current_value,
            'balance': self.portfolio_manager.current_balance,
            'positions': self.portfolio_manager.positions.copy(),
            'trades_executed': trades_executed,
            'episode_trades': self.episode_trades.copy(),
            'actions': action,
            'date': self.data.index[self.current_step],
            'current_data': self.data.iloc[self.current_step]
        }
        self._trade_history.append(info)

        return self._get_observation(), reward, done, False, info

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        self._initialize_state()
        self.portfolio_manager = PortfolioManager(self.initial_balance, self.transaction_cost)

        info = {
            'net_worth': self.initial_balance,
            'balance': self.portfolio_manager.current_balance,
            'positions': self.portfolio_manager.positions.copy(),
            'trades_executed': {symbol: False for symbol in self.symbols},
            'episode_trades': self.episode_trades,
            'actions': {},
            'date': self.data.index[self.current_step],
            'current_data': self.data.iloc[self.current_step]
        }

        return self._get_observation(), info

    def get_portfolio_history(self) -> List[float]:
        return self.portfolio_manager.portfolio_value_history