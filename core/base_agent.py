#!/usr/bin/env python
import logging
import numpy as np
import pandas as pd
import os
import warnings
from datetime import datetime, timedelta
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from typing import Dict, Any, Optional, List, Union, Tuple, cast
from numpy.typing import NDArray
from core.visualization import TradingVisualizer
from metrics.metrics_calculator import MetricsCalculator
from environment import TradingEnv
from data.data_handler import DataHandler
from utils.common import (type_check, MAX_POSITION_SIZE, MIN_POSITION_SIZE,
                          DEFAULT_STOP_LOSS, MIN_TRADE_SIZE)

# Suppress TF logging and CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


from core.config import DEFAULT_PPO_PARAMS, PARAM_RANGES, DEFAULT_POLICY_KWARGS

class UnifiedTradingAgent:
    """Unified trading agent combining all trading functionality."""

    @type_check
    def __init__(self,
                 optimize_for_sharpe: bool = True,
                 tensorboard_log: str = "./tensorboard_logs/",
                 seed: Optional[int] = None) -> None:
        """Initialize the unified trading agent."""
        self.env: Optional[xTradingEnv] = None
        self.model: Optional[PPO] = None
        self.data_handler = DataHandler()
        self.portfolio_history: List[float] = []
        self.positions_history: List[Dict[str, float]] = []
        self.portfolio_data: Dict[str, pd.DataFrame] = {}
        self.evaluation_metrics: Dict[str, Union[float, List[float], int]] = {
            'returns': [],
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'information_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'win_rate': 0.0
        }

        self.optimize_for_sharpe = optimize_for_sharpe
        self.tensorboard_log = tensorboard_log
        self.seed = seed
        self.ppo_params = self.DEFAULT_PPO_PARAMS.copy()

    @type_check
    def prepare_processed_data(self, stock_names: List, start_date: datetime,
                               end_date: datetime) -> pd.DataFrame:
        """Fetch and prepare data."""
        self.portfolio_data = self.data_handler.fetch_data(stock_names, start_date,
                                                           end_date)
        if not self.portfolio_data:
            raise ValueError("No data found in database")

        # Now we extract the features for the model:
        prepared_data = self.data_handler.prepare_data(self.portfolio_data)
        return next(iter(prepared_data.values()))

    @type_check
    def initialize_env(self, data: pd.DataFrame,
                       env_params: Dict[str, Any]) -> None:
        """Initialize trading environment."""
        env_params['min_transaction_size'] = MIN_TRADE_SIZE
        self.env = TradingEnv(
            data=data,
            initial_balance=env_params['initial_balance'],
            transaction_cost=env_params['transaction_cost'],
            use_position_profit=env_params.get('use_position_profit', False),
            use_holding_bonus=env_params.get('use_holding_bonus', False),
            use_trading_penalty=env_params.get('use_trading_penalty', False),
            training_mode=True)
        self.portfolio_history.clear()
        self.positions_history.clear()

    @type_check
    def configure_ppo(self,
                      ppo_params: Optional[Dict[str, Any]] = None) -> None:
        """Configure PPO model with validated parameters."""
        if not self.env:
            raise ValueError(
                "Environment not initialized. Call initialize_env first.")

        if ppo_params:
            for k, v in ppo_params.items():
                self.ppo_params[k] = v

        # Validate parameters
        for param, value in self.ppo_params.items():
            if param in self.PARAM_RANGES and value is not None:
                min_val, max_val = self.PARAM_RANGES[param]
                if not (min_val <= value <= max_val):
                    logger.warning(
                        f"Parameter {param} value {value} outside range [{min_val}, {max_val}]"
                    )
                    self.ppo_params[param] = max(min_val, min(value, max_val))

        # Set network architecture
        policy_kwargs = ({
            'net_arch': [dict(pi=[128, 128, 128], vf=[128, 128, 128])]
        } if self.optimize_for_sharpe else self.DEFAULT_POLICY_KWARGS.copy())

        if 'verbose' not in self.ppo_params:
            self.ppo_params['verbose'] = 1

        try:
            self.model = PPO("MlpPolicy",
                             self.env,
                             **self.ppo_params,
                             tensorboard_log=self.tensorboard_log,
                             policy_kwargs=policy_kwargs,
                             seed=self.seed)
        except Exception as e:
            logger.exception("Failed to initialize PPO model")
            raise

    @type_check
    def train(self,
              stock_names: List,
              start_date: datetime,
              end_date: datetime,
              env_params: Dict[str, Any],
              ppo_params: Dict[str, Any],
              callback: Optional[BaseCallback] = None) -> Dict[str, float]:
        """Train the agent."""
        data = self.prepare_processed_data(stock_names, start_date, end_date)
        self.initialize_env(data, env_params)
        self.configure_ppo(ppo_params)

        total_timesteps = max(1, (end_date - start_date).days)
        eval_callback = EvalCallback(self.env,
                                     best_model_save_path='./best_model/',
                                     log_path='./eval_logs/',
                                     eval_freq=1000,
                                     deterministic=True,
                                     render=False)

        try:
            self.model.learn(total_timesteps=total_timesteps,
                             callback=callback or eval_callback)
            self.save("trained_model.zip")

            self.portfolio_history = self.env.get_portfolio_history()
            if len(self.portfolio_history) > 1:
                returns = MetricsCalculator.calculate_returns(
                    self.portfolio_history)
                return {
                    'sharpe_ratio':
                    MetricsCalculator.calculate_sharpe_ratio(returns),
                    'max_drawdown':
                    MetricsCalculator.calculate_maximum_drawdown(
                        self.portfolio_history),
                    'sortino_ratio':
                    MetricsCalculator.calculate_sortino_ratio(returns),
                    'volatility':
                    MetricsCalculator.calculate_volatility(returns),
                    'total_return':
                    (self.portfolio_history[-1] - self.portfolio_history[0]) /
                    self.portfolio_history[0],
                    'final_value':
                    self.portfolio_history[-1]
                }
        except Exception as e:
            logger.exception("Error during training")
            raise

        return {}

    @type_check
    def predict(self,
                observation: NDArray,
                deterministic: bool = True) -> NDArray:
        """Generate trading action."""
        if self.model is None:
            raise ValueError("Model not initialized")

        action, _ = self.model.predict(observation,
                                       deterministic=deterministic)
        action_val = int(
            action.item() if isinstance(action, np.ndarray) else action)

        # if not 0 <= action_val <= 2:
        #     raise ValueError(f"Invalid action value: {action_val}")

        return np.array([action_val])

    @type_check
    def test(self, stock_names: List, start_date: datetime, end_date: datetime,
             env_params: Dict[str, Any]) -> Dict[str, Any]:
        """Test trained model."""
        data = self.prepare_processed_data(stock_names, start_date, end_date)
        self.initialize_env(data, env_params)
        self.load("trained_model.zip")

        # Ensure trade history is reset
        if hasattr(self.env, '_trade_history'):
            self.env._trade_history = []
            
        obs, _ = self.env.reset_portfolio_and_balance()
        done = False
        info_history = []

        while not done:
            action = self.predict(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            info_history.append(info)

        portfolio_history = self.env.get_portfolio_history()
        returns = MetricsCalculator.calculate_returns(portfolio_history)

        return {
            'portfolio_history':
            portfolio_history,
            'returns':
            returns,
            'info_history':
            info_history,
            'action_plot':
            TradingVisualizer().plot_discrete_actions(info_history),
            'combined_plot':
            TradingVisualizer().plot_actions_with_price(info_history, data),
            'metrics': {
                'sharpe_ratio':
                MetricsCalculator.calculate_sharpe_ratio(returns),
                'sortino_ratio':
                MetricsCalculator.calculate_sortino_ratio(returns),
                'information_ratio':
                MetricsCalculator.calculate_information_ratio(returns),
                'max_drawdown':
                MetricsCalculator.calculate_maximum_drawdown(
                    portfolio_history),
                'volatility':
                MetricsCalculator.calculate_volatility(returns)
            }
        }

    @type_check
    def update_state(self, portfolio_value: float,
                     positions: Dict[str, float]) -> None:
        """Update portfolio state."""
        if portfolio_value < 0:
            raise ValueError("Invalid portfolio value")

        for symbol, size in positions.items():
            if not np.isfinite(size):
                raise ValueError(f"Invalid position size for {symbol}")
            if abs(size) > MAX_POSITION_SIZE:
                raise ValueError(f"Position size exceeds limit for {symbol}")

        try:
            if self.env and hasattr(self.env, 'data'):
                current_prices = {
                    symbol: self.env.data.loc[self.env.current_step, 'Close']
                    for symbol in positions
                }
                calc_value = sum(size * current_prices[symbol]
                                 for symbol, size in positions.items())
                cash_balance = getattr(self.env, 'balance', 0.0)
                total_calc = calc_value + cash_balance

                if not np.isclose(total_calc, portfolio_value, rtol=1e-3):
                    logger.warning(
                        f"Portfolio value mismatch: calculated={total_calc:.2f}, reported={portfolio_value:.2f}"
                    )
        except Exception as e:
            logger.error(f"Error checking portfolio: {str(e)}")

        self.portfolio_history.append(portfolio_value)
        self.positions_history.append(positions)
        self._update_metrics()

    def _update_metrics(self) -> None:
        """Update performance metrics."""
        try:
            # Only calculate metrics if we have enough history
            if len(self.portfolio_history) > 1:
                returns = MetricsCalculator.calculate_returns(
                    self.portfolio_history)
                
                # Calculate metrics with error handling
                metrics_dict = {
                    'returns': float(np.mean(returns)) if len(returns) > 0 else 0.0,
                    'sharpe_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'information_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'total_trades': 0,
                    'win_rate': 0.0
                }
                
                # Only calculate ratios if we have valid returns
                if len(returns) > 0:
                    try:
                        metrics_dict['sharpe_ratio'] = MetricsCalculator.calculate_sharpe_ratio(returns)
                        metrics_dict['sortino_ratio'] = MetricsCalculator.calculate_sortino_ratio(returns)
                        metrics_dict['information_ratio'] = MetricsCalculator.calculate_information_ratio(returns)
                        metrics_dict['max_drawdown'] = MetricsCalculator.calculate_maximum_drawdown(self.portfolio_history)
                    except Exception as e:
                        logger.warning(f"Error calculating some metrics: {str(e)}")
                
                # Calculate trade metrics
                valid_positions = [p for p in self.positions_history if isinstance(p, dict) and p]
                metrics_dict['total_trades'] = len(valid_positions)
                
                if len(valid_positions) > 1:
                    try:
                        profitable_trades = sum(1 for i in range(1, len(valid_positions))
                                              if sum(valid_positions[i].values()) > sum(valid_positions[i-1].values()))
                        metrics_dict['win_rate'] = profitable_trades / (len(valid_positions) - 1)
                    except Exception as e:
                        logger.warning(f"Error calculating trade metrics: {str(e)}")
                
                # Update the metrics
                self.evaluation_metrics.update(metrics_dict)
            
        except Exception as e:
            logger.exception("Critical error updating metrics")
            # Keep existing metrics instead of resetting to 0
            logger.warning("Keeping previous metrics due to update failure")

    @type_check
    def get_metrics(self) -> Dict[str, Union[float, List[float], int]]:
        """Get current metrics."""
        return self.evaluation_metrics.copy()

    @type_check
    def save(self, path: str) -> None:
        """Save model."""
        if not path.strip():
            raise ValueError("Empty path")
        if not self.model:
            raise ValueError("Model not initialized")
        self.model.save(path)

    @type_check
    def load(self, path: str) -> None:
        """Load model."""
        if not path.strip():
            raise ValueError("Empty path")
        if not self.env:
            raise ValueError("Environment not initialized")
        self.model = PPO.load(path, env=self.env)
