
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
from core.portfolio_manager import PortfolioManager
import streamlit as st

from core.config import DEFAULT_PPO_PARAMS, PARAM_RANGES, DEFAULT_POLICY_KWARGS
from utils.common import (type_check, MAX_POSITION_SIZE, MIN_POSITION_SIZE,
                          DEFAULT_STOP_LOSS, MIN_TRADE_SIZE)

# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class UnifiedTradingAgent:
    """Unified trading agent with modular components for trading functionality."""

    @type_check
    def __init__(self,
                 optimize_for_sharpe: bool = True,
                 tensorboard_log: str = "./tensorboard_logs/",
                 seed: Optional[int] = None) -> None:
        """Initialize the unified trading agent with configuration."""
        self._init_state()
        self.optimize_for_sharpe = optimize_for_sharpe
        self.tensorboard_log = tensorboard_log
        self.seed = seed
        self.ppo_params = DEFAULT_PPO_PARAMS.copy()

    def _init_state(self) -> None:
        """Initialize agent state variables."""
        self.env = None
        self.model = None
        self.stocks_data = {}
        self.portfolio_manager = None
        self.evaluation_metrics = {
            'returns': [],
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'information_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'win_rate': 0.0
        }

    @type_check
    def prepare_processed_data(self, stock_names: List, start_date: datetime,
                             end_date: datetime) -> pd.DataFrame:
        """Prepare and validate trading data."""
        self.stocks_data = st.session_state.data_handler.fetch_data(stock_names, start_date,
                                                                  end_date)
        if isinstance(self.stocks_data, pd.DataFrame) and self.stocks_data.empty:
            raise ValueError("No data found in database")
        
        if not any(f'Close_{symbol}' in self.stocks_data.columns for symbol in stock_names):
            raise ValueError("Data format incorrect - missing Close_SYMBOL columns")
            
        prepared_data = self.stocks_data #st.session_state.data_handler.prepare_data(self.stocks_data)
        return prepared_data

    @type_check
    def initialize_env(self, data: pd.DataFrame,
                      env_params: Dict[str, Any]) -> None:
        """Initialize trading environment with parameters."""
        env_params['min_transaction_size'] = MIN_TRADE_SIZE
        self.env = TradingEnv(
            data=data,
            initial_balance=env_params['initial_balance'],
            transaction_cost=env_params['transaction_cost'],
            use_position_profit=env_params.get('use_position_profit', False),
            use_holding_bonus=env_params.get('use_holding_bonus', False),
            use_trading_penalty=env_params.get('use_trading_penalty', False),
            training_mode=True)
        
        # Use environment's portfolio manager instead of creating a new one
        self.portfolio_manager = self.env.portfolio_manager

    @type_check
    def configure_ppo(self, ppo_params: Optional[Dict[str, Any]] = None) -> None:
        """Configure PPO model with validated parameters."""
        if not self.env:
            raise ValueError("Environment not initialized. Call initialize_env first.")

        if ppo_params:
            self.ppo_params.update(ppo_params)

        self._validate_ppo_params()
        self._setup_ppo_model()

    def _validate_ppo_params(self) -> None:
        """Validate PPO parameters against defined ranges."""
        for param, value in self.ppo_params.items():
            if param in PARAM_RANGES and value is not None:
                min_val, max_val = PARAM_RANGES[param]
                if not (min_val <= value <= max_val):
                    logger.warning(
                        f"Parameter {param} value {value} outside range [{min_val}, {max_val}]"
                    )
                    self.ppo_params[param] = max(min_val, min(value, max_val))

    def _setup_ppo_model(self) -> None:
        """Set up PPO model with current configuration."""
        policy_kwargs = ({
            'net_arch': [dict(pi=[128, 128, 128], vf=[128, 128, 128])]
        } if self.optimize_for_sharpe else DEFAULT_POLICY_KWARGS.copy())

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
        """Train the agent with progress logging."""
        logger.info(f"Starting training with {len(stock_names)} stocks from {start_date} to {end_date}")
        data = self.prepare_processed_data(stock_names, start_date, end_date)
        logger.info(f"Prepared data shape: {data.shape}")
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
            return self._calculate_training_metrics()
        except Exception as e:
            logger.exception("Error during training")
            raise

    def _calculate_training_metrics(self) -> Dict[str, float]:
        """Get training metrics directly from portfolio manager."""
        if not self.env or not self.env.portfolio_manager:
            return {}
        return self.env.portfolio_manager.get_portfolio_metrics()

    @type_check
    def predict(self,
                observation: NDArray,
                deterministic: bool = True) -> NDArray:
        """Generate trading action with input validation."""
        if self.model is None:
            raise ValueError("Model not initialized")

        action, _ = self.model.predict(observation, deterministic=deterministic)
        
        if isinstance(action, np.ndarray):
            if action.size == 1:
                return np.array([int(action.item())])
            return action.astype(int)
        return np.array([int(action)])

    @type_check
    def test(self, stock_names: List, start_date: datetime, end_date: datetime,
             env_params: Dict[str, Any]) -> Dict[str, Any]:
        """Test trained model with comprehensive metrics."""
        data = self.prepare_processed_data(stock_names, start_date, end_date)
        self.initialize_env(data, env_params)
        self.load("trained_model.zip")

        if hasattr(self.env, '_trade_history'):
            self.env._trade_history = []

        obs, info = self.env.reset()
        logger.info(f"Observation initial test: {obs}")
        
        done = False
        info_history = []
        
        while not done:
            action = self.predict(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            info_history.append(info)

        return self._prepare_test_results(info_history, data)

    def _prepare_test_results(self, info_history: List[Dict], 
                            data: pd.DataFrame) -> Dict[str, Any]:
        """Prepare comprehensive test results."""
        portfolio_history = self.env.get_portfolio_history()
        returns = MetricsCalculator.calculate_returns(portfolio_history)

        return {
            'portfolio_history': portfolio_history,
            'returns': returns,
            'info_history': info_history,
            'action_plot': TradingVisualizer().plot_discrete_actions(info_history),
            'combined_plot': TradingVisualizer().plot_actions_with_price(info_history, data),
            'metrics': {
                'sharpe_ratio': MetricsCalculator.calculate_sharpe_ratio(returns),
                'sortino_ratio': MetricsCalculator.calculate_sortino_ratio(returns),
                'information_ratio': MetricsCalculator.calculate_information_ratio(returns),
                'max_drawdown': MetricsCalculator.calculate_maximum_drawdown(portfolio_history),
                'volatility': MetricsCalculator.calculate_volatility(returns)
            }
        }

    

    def _update_metrics(self) -> None:
        """Update performance metrics with error handling."""
        try:
            if len(self.portfolio_history) > 1:
                returns = MetricsCalculator.calculate_returns(self.portfolio_history)
                self._calculate_and_update_metrics(returns)
        except Exception as e:
            logger.exception("Critical error updating metrics")

    def _calculate_and_update_metrics(self, returns: List[float]) -> None:
        """Calculate and update all metrics."""
        metrics_dict = {
            'returns': float(np.mean(returns)) if returns else 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'information_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'win_rate': 0.0
        }

        if returns:
            try:
                metrics_dict.update({
                    'sharpe_ratio': MetricsCalculator.calculate_sharpe_ratio(returns),
                    'sortino_ratio': MetricsCalculator.calculate_sortino_ratio(returns),
                    'information_ratio': MetricsCalculator.calculate_information_ratio(returns),
                    'max_drawdown': MetricsCalculator.calculate_maximum_drawdown(self.portfolio_history)
                })
            except Exception as e:
                logger.warning(f"Error calculating some metrics: {str(e)}")

        valid_positions = [p for p in self.positions_history if isinstance(p, dict) and p]
        metrics_dict['total_trades'] = len(valid_positions)

        if len(valid_positions) > 1:
            try:
                profitable_trades = sum(1 for i in range(1, len(valid_positions))
                                     if sum(valid_positions[i].values()) > sum(valid_positions[i-1].values()))
                metrics_dict['win_rate'] = profitable_trades / (len(valid_positions) - 1)
            except Exception as e:
                logger.warning(f"Error calculating trade metrics: {str(e)}")

        self.evaluation_metrics.update(metrics_dict)

    @type_check
    def get_metrics(self) -> Dict[str, Union[float, List[float], int]]:
        """Get current metrics."""
        return self.evaluation_metrics.copy()

    @type_check
    def save(self, path: str) -> None:
        """Save model with validation."""
        if not path.strip():
            raise ValueError("Empty path")
        if not self.model:
            raise ValueError("Model not initialized")
        self.model.save(path)

    @type_check
    def load(self, path: str) -> None:
        """Load model with validation."""
        if not path.strip():
            raise ValueError("Empty path")
        if not self.env:
            raise ValueError("Environment not initialized")
        self.model = PPO.load(path, env=self.env)
