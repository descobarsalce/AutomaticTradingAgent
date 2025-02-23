from datetime import datetime
import gymnasium as gym
import streamlit as st
from metrics.metrics_calculator import MetricsCalculator
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

async def fetch_trading_data(stock_names: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch and validate trading data with improved error handling."""
    try:
        data = await st.session_state.data_handler.fetch_data(stock_names, start_date, end_date)
        if data.empty:
            raise ValueError(f"No data available for symbols {stock_names} between {start_date} and {end_date}")

        # Validate that we have data for all symbols
        symbols_with_data = set(col.split('_')[1] for col in data.columns if '_' in col)
        missing_symbols = set(stock_names) - symbols_with_data
        if missing_symbols:
            raise ValueError(f"Missing data for symbols: {missing_symbols}")

        # Ensure we have all required columns for each symbol
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for symbol in stock_names:
            missing_cols = [col for col in required_cols if f"{col}_{symbol}" not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing columns {missing_cols} for symbol {symbol}")

        return data
    except Exception as e:
        logger.error(f"Error fetching trading data: {str(e)}")
        raise ValueError(f"Failed to fetch valid trading data: {str(e)}")

class TradingEnv(gym.Env):
    def __init__(self,
                 stock_names: List[str],
                 start_date: datetime,
                 end_date: datetime,
                 initial_balance: float = 10000,
                 transaction_cost: float = 0.0,
                 max_pct_position_by_asset: float = 0.2,
                 use_position_profit: bool = False,
                 use_holding_bonus: bool = False,
                 use_trading_penalty: bool = False,
                 training_mode: bool = True,
                 observation_days: int = 2,
                 burn_in_days: int = 20):  # New parameter for burn-in period.
        # ...existing code to fetch data and validate parameters...
        data = fetch_trading_data(stock_names, start_date, end_date)
        # New: Preprocess data: handle missing values and normalize
        # data = preprocess_data(data)
        self._validate_init_params(data, initial_balance, transaction_cost, max_pct_position_by_asset)
        self._full_data = data.copy()
        self.data = data
        self.max_pct_position_by_asset = max_pct_position_by_asset
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.stock_names = stock_names

        # Set reward shaping flags
        self.use_position_profit = use_position_profit
        self.use_holding_bonus = use_holding_bonus
        self.use_trading_penalty = use_trading_penalty
        self.training_mode = training_mode

        # Initialize portfolio manager before constructing observation space.
        self.portfolio_manager = PortfolioManager(initial_balance, transaction_cost, price_fetcher=self._get_current_price)

        # initialize state & observation space.
        self.observation_days = observation_days  # Store the number of days to keep in the observation
        self.burn_in_days = burn_in_days
        self._initialize_state()
        self.observation_space = self._create_observation_space()  # Ensure observation space is created before use

        # Set the action space to Box with shape (n_stocks,) and range [-1, 1]
        self.action_space = spaces.Box(
            low=-1,
            high=1, 
            shape=(len(stock_names),),
            dtype=np.float32
        )

    def _validate_init_params(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                            initial_balance: float, transaction_cost: float,
                            max_pct_position_by_asset: float) -> None:
        if initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        if transaction_cost < 0:
            raise ValueError("Transaction cost cannot be negative")
        if not 0 < max_pct_position_by_asset <= 1:
            raise ValueError("Position size must be between 0 and 1")
        if isinstance(data, pd.DataFrame) and data.empty:
            raise ValueError("Empty data provided")

    def _initialize_state(self) -> None:
        self.current_step = 0
        self.episode_trades = {symbol: 0 for symbol in self.stock_names}
        self.observation_history: List[np.ndarray] = []  # New observation history

    # --- Auxiliary functions to reduce repetition ---
    def _get_current_price(self, symbol: str) -> float:
        """Retrieve the current closing price for a symbol using _full_data."""
        return float(self._full_data.iloc[self.current_step][f'Close_{symbol}'])

    def _get_current_data(self) -> pd.Series:
        """Retrieve the current data row."""
        return self.data.iloc[self.current_step]
    # --- End Auxiliary functions ---

    def _create_observation_space(self) -> spaces.Box:
        # Define a fixed observation space shape based on observation_days.
        n_features = len(self.stock_names) * 2 + 1
        obs_dim = self.observation_days * n_features
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _construct_observation(self) -> np.ndarray:
        """Collect current prices, positions, and balance."""
        # Construct current observation
        current_obs = []
        for symbol in self.stock_names:
            current_price = self._get_current_price(symbol)
            position = self.portfolio_manager.positions.get(symbol, 0.0)
            current_obs.extend([current_price, position])
        current_obs.append(self.portfolio_manager.current_balance)
        current_obs = np.array(current_obs, dtype=np.float32)

        # Log immediate state
        logger.debug(f"Current step observation: {current_obs}")

        # Add to history
        self.observation_history.append(current_obs)

        if len(self.observation_history) < self.observation_days:
            missing = self.observation_days - len(self.observation_history)
            pad = [self.observation_history[0]] * missing if self.observation_history else [np.zeros(self.observation_space.shape[0] // self.observation_days, dtype=np.float32)] * missing
            history = pad + self.observation_history
        else:
            history = self.observation_history[-self.observation_days:]
        combined_obs = np.concatenate(history, axis=0)

        # New: Check and replace NaNs in the observation.
        if np.isnan(combined_obs).any():
            logger.warning("NaN values detected in observation; replacing with zeros.")
            combined_obs = np.nan_to_num(combined_obs)

        # Log full observation with history and current observation separately
        logger.info(f"Full observation (including history):\nShape: {combined_obs.shape}\nData: {combined_obs}")
        logger.info(f"Current observation (this date):\nShape: {current_obs.shape}\nData: {current_obs}")

        return combined_obs

    def _compute_reward(self, trades_executed: Dict[str, bool]) -> float:
        # Use the portfolio manager's history for incremental reward calculation.
        history = self.portfolio_manager.portfolio_value_history
        if len(history) > 1:
            # Compute step-specific return based on the last two recorded portfolio values.
            last_value = history[-2]
            current_value = history[-1]
            if last_value > 0:
                step_return = (current_value - last_value) #/ last_value
            else:
                step_return = 0.0
        else:
            step_return = 0.0

        reward = step_return

        # Risk management adjustments
        max_drawdown = MetricsCalculator.calculate_maximum_drawdown(history)
        reward -= max_drawdown

        returns = MetricsCalculator.calculate_returns(history)
        volatility = MetricsCalculator.calculate_volatility(returns)
        reward -= volatility

        # Transaction cost penalty
        transaction_cost_penalty = sum(trades_executed.values()) * self.transaction_cost
        reward -= transaction_cost_penalty

        # Clip reward to prevent extreme values
        reward = np.clip(reward, -10, 10)

        # Check for NaN values in reward
        if np.isnan(reward):
            logger.error(f"NaN detected in reward: {reward}")
            raise ValueError("NaN detected in reward")

        logger.debug(f"Computed reward (step_return based): {reward}")
        return reward

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        try:
            timestamp = self.data.index[self.current_step]
            trades_executed = self.portfolio_manager.execute_all_trades(
                self.stock_names,
                np.array(action, dtype=np.float32),
                self._get_current_price,
                self.max_pct_position_by_asset,
                timestamp
            )

            # Merge rejected BUY trade info into one log message.
            actions_arr = np.array(action, dtype=np.float32)
            rejected = [f"{symbol} (Action: {actions_arr[idx]}, Price: {self._get_current_price(symbol)})"
                        for idx, symbol in enumerate(self.stock_names)
                        if actions_arr[idx] > 0 and not trades_executed.get(symbol, False)]
            if rejected:
                logger.info(f"Rejected BUY trades for: {', '.join(rejected)}")

            # Calculate reward and update environment state.
            reward = self._compute_reward(trades_executed)
            self.current_step += 1
            obs = self._construct_observation()
            done = self.current_step >= len(self.data) - 1

            # Log one structured table summarizing the step.
            log_table = (
                "+-------------------------+----------------------------------------------+\n"
                f"| {'Step':<23} | {self.current_step!s:<44} |\n"
                f"| {'Timestamp':<23} | {timestamp!s:<44} |\n"
                f"| {'Balance':<23} | {self.portfolio_manager.current_balance!s:<44} |\n"
                f"| {'Positions':<23} | {str(self.portfolio_manager.positions):<44} |\n"
                f"| {'Actions':<23} | {str(action):<44} |\n"
                f"| {'Trades Exec':<23} | {str(trades_executed):<44} |\n"
                f"| {'Reward':<23} | {reward!s:<44} |\n"
                f"| {'Total Value':<23} | {self.portfolio_manager.get_total_value()!s:<44} |\n"
                "+-------------------------+----------------------------------------------+"
            )
            logger.info(log_table)

            info = {
                'net_worth': self.portfolio_manager.get_total_value(),
                'balance': self.portfolio_manager.current_balance,
                'positions': self.portfolio_manager.positions.copy(),
                'trades_executed': trades_executed,
                'episode_trades': self.episode_trades.copy(),
                'actions': action,
                'date': timestamp,
                'current_data': self._get_current_data(),
                'portfolio_value': self.portfolio_manager.get_total_value(),
                'step': self.current_step
            }
            return obs, reward, done, False, info

        except Exception as e:
            logger.error(f"Error in step method: {str(e)}")
            return self._construct_observation(), 0.0, True, False, {
                'error': str(e),
                'net_worth': self.portfolio_manager.get_total_value(),
                'balance': self.portfolio_manager.current_balance,
                'positions': self.portfolio_manager.positions.copy(),
                'trades_executed': {},
                'episode_trades': self.episode_trades.copy(),
                'actions': action,
                'date': self.data.index[self.current_step-1] if self.current_step > 0 else self.data.index[0],
                'current_data': self._get_current_data(),
                'portfolio_value': self.portfolio_manager.get_total_value(),
                'step': self.current_step
            }

    def _verify_observation_consistency(self) -> None:
        """Verify observation history consistency"""
        if len(self.observation_history) == 0:
            raise ValueError("Observation history is empty")

        # Verify shape consistency
        first_obs_shape = self.observation_history[0].shape
        for idx, obs in enumerate(self.observation_history):
            if obs.shape != first_obs_shape:
                raise ValueError(f"Inconsistent observation shape at index {idx}")

        # Verify no NaN values
        if any(np.isnan(obs).any() for obs in self.observation_history):
            raise ValueError("NaN values detected in observation history")

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        self._initialize_state()
        self.portfolio_manager = PortfolioManager(self.initial_balance, self.transaction_cost)
        self.portfolio_manager.portfolio_value_history = [self.initial_balance]
        self.current_step = 0
        self.observation_history.clear()

        # Burn-in period: automatically take no-op action (assumed to be action "0")
        burn_in_counter = 0
        while burn_in_counter < self.burn_in_days and not self._burn_in_done():
            obs, reward, done, _, info = self.step(np.zeros(len(self.stock_names)))
            burn_in_counter += 1
            if done:
                break

        combined_obs = self._construct_observation()
        info = {
            'net_worth': self.initial_balance,
            'balance': self.portfolio_manager.current_balance,
            'positions': self.portfolio_manager.positions.copy(),
            'trades_executed': {symbol: False for symbol in self.stock_names},
            'episode_trades': self.episode_trades,
            'actions': {},
            'date': self.data.index[self.current_step],
            'current_data': self._get_current_data()
        }

        logger.debug(f"Reset info: {info}")
        return combined_obs, info

    def _burn_in_done(self) -> bool:
        return len(self.observation_history) >= self.observation_days

    def verify_env_state(self) -> Dict[str, Any]:
        """Verify environment state consistency"""
        state = {
            'current_step': self.current_step,
            'observation_shapes': [obs.shape for obs in self.observation_history],
            'has_nans': any(np.isnan(obs).any() for obs in self.observation_history),
            'portfolio_balance': self.portfolio_manager.current_balance,
            'data_remaining': len(self._full_data) - self.current_step
        }
        return state

    def get_portfolio_history(self) -> List[float]:
        return self.portfolio_manager.portfolio_value_history

    def _prepare_test_results(self, info_history: List[Dict]) -> Dict[str, Any]:
        portfolio_history = self.portfolio_manager.portfolio_value_history
        returns = MetricsCalculator.calculate_returns(portfolio_history)
        # ...existing code...
        result = {
            # ...existing code...
            'metrics': {
                'sharpe_ratio': float(MetricsCalculator.calculate_sharpe_ratio(returns)),
                'sortino_ratio': float(MetricsCalculator.calculate_sortino_ratio(returns)),
                'information_ratio': float(MetricsCalculator.calculate_information_ratio(returns)),
                'max_drawdown': float(MetricsCalculator.calculate_maximum_drawdown(portfolio_history)),
                'volatility': float(MetricsCalculator.calculate_volatility(returns))
            }
        }
        return result