#!/usr/bin/env python
import logging
import numpy as np
import pandas as pd
import os
import warnings
from datetime import datetime, timedelta
from gymnasium import Env
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from typing import Dict, Any, Optional, List, Union, Tuple, cast
from numpy.typing import NDArray
from core.visualization import TradingVisualizer
from metrics.metrics_calculator import MetricsCalculator
from environment import TradingEnv
from core.portfolio_manager import PortfolioManager
from metrics.metric_sink import MetricsSink, MetricsSinkConfig
from utils.data_splitter import TemporalDataSplitter
from environment.trading_env import fetch_trading_data

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


class MetricStreamingEvalCallback(BaseCallback):
    """Periodic validation episodes that stream metrics to a sink."""

    def __init__(self, eval_env: Env, metrics_sink: MetricsSink,
                 visualizer: TradingVisualizer, eval_freq: int = 1000,
                 seeds: Optional[List[int]] = None,
                 log_dir: str = "metrics/eval") -> None:
        super().__init__()
        self.eval_env = eval_env
        self.metrics_sink = metrics_sink
        self.visualizer = visualizer
        self.eval_freq = eval_freq
        self.seeds = seeds or [0]
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        for seed in self.seeds:
            self._run_validation_episode(seed)
        return True

    def _run_validation_episode(self, seed: int) -> None:
        obs, info = self.eval_env.reset(seed=seed)
        done = False
        info_history: List[Dict[str, Any]] = []

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = terminated or truncated
            info_history.append(info)

        self._log_results(info_history)

    def _log_results(self, info_history: List[Dict[str, Any]]) -> None:
        portfolio_history = self.eval_env.get_portfolio_history()
        returns = MetricsCalculator.calculate_returns(portfolio_history)
        pm = self.eval_env.portfolio_manager

        metrics_payload = {
            'pnl': pm.get_total_value() - pm.initial_balance,
            'max_drawdown': MetricsCalculator.calculate_maximum_drawdown(portfolio_history),
            'turnover': pm.calculate_turnover(),
            'sharpe_ratio': MetricsCalculator.calculate_sharpe_ratio(returns),
            'sortino_ratio': MetricsCalculator.calculate_sortino_ratio(returns),
            'information_ratio': MetricsCalculator.calculate_information_ratio(returns),
            'portfolio': {
                'positions': pm.positions.copy(),
                'balance': pm.current_balance,
                'total_value': pm.get_total_value(),
            }
        }

        if info_history and self.eval_env.data is not None:
            last_date = info_history[-1].get('date', datetime.utcnow())
            filename = f"eval_actions_{last_date.strftime('%Y%m%d_%H%M%S')}"
            snapshot_path = os.path.join(self.log_dir, f"{filename}.html")
            metrics_payload['action_snapshot'] = self.visualizer.snapshot_actions_with_price(
                info_history, self.eval_env.data, snapshot_path)

        self.metrics_sink.emit("validation_episode", metrics_payload)


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
        start_time = datetime.now()
        logger.info("🤖 Initializing trading agent state...")
        logger.info("🔄 Setting up environment variables...")
        self.env = None
        self.model = None
        logger.info(f"Agent state initialization completed in {(datetime.now() - start_time).total_seconds():.2f}s")
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

    def _build_training_schedule(self, data_points: int,
                                 schedule_config: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        config = {
            'episode_length': 256,
            'epochs': 3,
            'warmup_fraction': 0.1,
        }
        if schedule_config:
            config.update(schedule_config)

        episode_length = max(1, min(config['episode_length'], data_points))
        warmup_steps = int(data_points * config['warmup_fraction'])
        total_timesteps = warmup_steps + (episode_length * config['epochs'])

        return {
            'episode_length': episode_length,
            'warmup_steps': warmup_steps,
            'total_timesteps': max(total_timesteps, data_points)
        }

    def _get_data_index(self, stock_names: List[str], start_date: datetime,
                        end_date: datetime):
        try:
            data = fetch_trading_data(stock_names, start_date, end_date)
            return data.index
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Falling back to naive date split; unable to fetch data index for split: %s",
                exc,
            )
            return None

    @type_check
    def initialize_env(self, stock_names: List[str], start_date: datetime,
                      end_date: datetime, env_params: Dict[str, Any],
                      feature_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize trading environment with parameters.

        Args:
            stock_names: List of stock symbols
            start_date: Training start date
            end_date: Training end date
            env_params: Environment parameters (balance, costs, etc.)
            feature_config: Optional feature engineering configuration
        """
        logger.info(f"Initializing environment with params: {env_params}")

        # Clean up env_params to remove any deprecated parameters
        cleaned_params = {
            'initial_balance': env_params.get('initial_balance', 10000),
            'transaction_cost': env_params.get('transaction_cost', 0.0),
            'max_pct_position_by_asset': env_params.get('max_pct_position_by_asset', 0.2),
            'use_position_profit': env_params.get('use_position_profit', False),
            'use_holding_bonus': env_params.get('use_holding_bonus', False),
            'use_trading_penalty': env_params.get('use_trading_penalty', False),
            'observation_days': env_params.get('history_length', 3)  # Convert history_length to observation_days
        }

        # Remove any None values
        cleaned_params = {k: v for k, v in cleaned_params.items() if v is not None}

        logger.info(f"Cleaned environment parameters: {cleaned_params}")

        # Log a centralized table of environment parameters
        headers = f"{'Parameter':<25}{'Value':<15}"
        rows = "\n".join([f"{k:<25}{v!s:<15}" for k, v in cleaned_params.items()])
        logger.info("\nEnvironment Parameters:\n" + headers + "\n" + rows)

        # Log feature configuration
        if feature_config:
            logger.info(f"Feature engineering enabled: {feature_config.get('use_feature_engineering', False)}")

        try:
            # Initializes the TradingEnv which uses PortfolioManager with balance check in execute_trade
            self.env = TradingEnv(
                stock_names=stock_names,
                start_date=start_date,
                end_date=end_date,
                **cleaned_params,
                feature_config=feature_config,
                training_mode=True
            )
            logger.info("Environment initialized successfully")
            self.portfolio_manager = self.env.portfolio_manager
        except Exception as e:
            logger.error(f"Failed to initialize environment: {str(e)}")
            raise

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
        policy_kwargs = ({'net_arch': [dict(pi=[128, 128, 128], vf=[128, 128, 128])]} 
                         if self.optimize_for_sharpe 
                         else DEFAULT_POLICY_KWARGS.copy())

        if 'verbose' not in self.ppo_params:
            self.ppo_params['verbose'] = 1

        # Example of adding entropy regularization
        self.ppo_params['ent_coef'] = 0.01  # Entropy coefficient

        try:
            # Suppress torch warning about class registration
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self.model = PPO("MlpPolicy",
                               self.env,
                               **self.ppo_params,
                           tensorboard_log=self.tensorboard_log,
                           policy_kwargs=policy_kwargs,
                           seed=self.seed)
        except Exception as e:
            logger.exception("Failed to initialize PPO model")
            raise

    def _init_env_and_model(self, stock_names: List, start_date: datetime, end_date: datetime,
                            env_params: Dict[str, Any], ppo_params: Dict[str, Any],
                            feature_config: Optional[Dict[str, Any]] = None):
        """Prepare environment and configure PPO model."""
        self.initialize_env(stock_names, start_date, end_date, env_params, feature_config)
        # Set default action space if not provided by the environment
        if self.env.action_space is None:
            import gymnasium as gym
            self.env.action_space = gym.spaces.Box(
                low=-1,
                high=1,
                shape=(len(stock_names),),
                dtype=np.float32
            )
        self.configure_ppo(ppo_params)

    def _create_validation_env(self, stock_names: List[str], date_range: Tuple[datetime, datetime],
                               env_params: Dict[str, Any],
                               feature_config: Optional[Dict[str, Any]]) -> TradingEnv:
        return TradingEnv(
            stock_names=stock_names,
            start_date=date_range[0],
            end_date=date_range[1],
            initial_balance=env_params.get('initial_balance', 10000),
            transaction_cost=env_params.get('transaction_cost', 0.0),
            max_pct_position_by_asset=env_params.get('max_pct_position_by_asset', 0.2),
            use_position_profit=env_params.get('use_position_profit', False),
            use_holding_bonus=env_params.get('use_holding_bonus', False),
            use_trading_penalty=env_params.get('use_trading_penalty', False),
            observation_days=env_params.get('history_length', 3),
            feature_config=feature_config,
            training_mode=False
        )
    
    def _calculate_training_metrics(self) -> Dict[str, float]:
        if not self.env or not self.env.portfolio_manager:
            return {}
        metrics = self.env.portfolio_manager.get_portfolio_metrics()
        # Make sure volatility is float
        metrics['volatility'] = float(metrics.get('volatility', 0.0))
        return metrics
    
    @type_check
    def train(self,
              stock_names: List,
              start_date: datetime,
              end_date: datetime,
              env_params: Dict[str, Any],
              ppo_params: Dict[str, Any],
              callback: Optional[BaseCallback] = None,
              feature_config: Optional[Dict[str, Any]] = None,
              validation_split: float = 0.2,
              schedule_config: Optional[Dict[str, Any]] = None,
              eval_freq: int = 1000,
              eval_seeds: Optional[List[int]] = None) -> Dict[str, float]:
        """Train the agent with progress logging.

        Args:
            stock_names: List of stock symbols to trade
            start_date: Training start date
            end_date: Training end date
            env_params: Environment parameters
            ppo_params: PPO hyperparameters
            callback: Optional training callback
            feature_config: Optional feature engineering configuration
            validation_split: Fraction of data to reserve for validation episodes
            schedule_config: Optional configuration for episode length and warmup
            eval_freq: Steps between validation episodes
            eval_seeds: Fixed seeds for validation rollouts

        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Starting training with {len(stock_names)} stocks from {start_date} to {end_date}")
        splitter = TemporalDataSplitter(validation_fraction=validation_split)
        date_index = self._get_data_index(stock_names, start_date, end_date)
        train_range, val_range = splitter.split_dates(start_date, end_date, date_index)
        logger.info(f"Train window: {train_range[0]} to {train_range[1]}")
        logger.info(f"Validation window: {val_range[0]} to {val_range[1]}")

        self._init_env_and_model(stock_names, train_range[0], train_range[1], env_params, ppo_params, feature_config)

        schedule = self._build_training_schedule(len(self.env.data), schedule_config)
        logger.info(f"Training schedule: {schedule}")

        validation_env = self._create_validation_env(stock_names, val_range, env_params, feature_config)
        metrics_sink = MetricsSink(MetricsSinkConfig())
        visualizer = TradingVisualizer()
        eval_callback = MetricStreamingEvalCallback(
            validation_env,
            metrics_sink,
            visualizer,
            eval_freq=eval_freq,
            seeds=eval_seeds or [7, 21]
        )

        callbacks: List[BaseCallback] = [eval_callback]
        if callback:
            callbacks.insert(0, callback)
        callback_to_use: BaseCallback = CallbackList(callbacks) if len(callbacks) > 1 else callbacks[0]

        try:
            self.model.learn(total_timesteps=schedule['total_timesteps'],
                           callback=callback_to_use)
            self.save("trained_model.zip")
            return self._calculate_training_metrics()
        except Exception as e:
            logger.exception("Error during training")
            raise

    @type_check
    def predict(self, observation: NDArray, deterministic: bool = True) -> NDArray:
        logger.info(f"\nGenerating prediction")
        logger.info(f"Input observation: {observation}")
        
        if self.model is None:
            logger.error("Model not initialized")
            raise ValueError("Model not initialized")
            
        action, _ = self.model.predict(observation, deterministic=deterministic)
        action = np.array(action, dtype=np.float32)
        logger.info(f"Generated action: {action}")
        return action

    @type_check
    def test(self, stock_names: List, start_date: datetime, end_date: datetime,
             env_params: Dict[str, Any], feature_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info(f"\n{'='*50}\nStarting test run\n{'='*50}")
        logger.info(f"Stocks: {stock_names}")
        logger.info(f"Period: {start_date} to {end_date}")

        self.initialize_env(stock_names, start_date, end_date, env_params, feature_config)
        self.load("trained_model.zip")

        if hasattr(self.env, '_trade_history'):
            self.env._trade_history = []

        obs, info = self.env.reset()
        logger.info(f"After reset - Initial balance: {self.env.portfolio_manager.current_balance}")
        logger.info(f"After reset - Initial positions: {self.env.portfolio_manager.positions}")
        logger.info(f"Initial observation: {obs}")
        
        done = False
        info_history = []
        
        while not done:
            action = self.predict(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            info_history.append(info)

        logger.info(f"Test run completed - Final positions: {self.env.portfolio_manager.positions}")
        return self._prepare_test_results(info_history)

    def _prepare_test_results(self, info_history: List[Dict]) -> Dict[str, Any]:
        """Prepare comprehensive test results."""
        portfolio_history = self.env.get_portfolio_history()
        returns = MetricsCalculator.calculate_returns(portfolio_history)

        return {
            'portfolio_history': portfolio_history,
            'returns': returns,
            'info_history': info_history,
            'action_plot': TradingVisualizer().plot_discrete_actions(info_history),
            'combined_plot': TradingVisualizer().plot_actions_with_price(info_history, self.env.data),
            'metrics': {
                'total_return': (self.env.portfolio_manager.get_total_value() - self.env.portfolio_manager.initial_balance) / self.env.portfolio_manager.initial_balance,
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
