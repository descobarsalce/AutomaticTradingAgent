#!/usr/bin/env python
import logging
import numpy as np
import pandas as pd
import os
import random
import warnings
from datetime import datetime, timedelta
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (BaseCallback, CallbackList,
                                                CheckpointCallback,
                                                EvalCallback)
from typing import Dict, Any, Optional, List, Union, Tuple, cast, TYPE_CHECKING
from numpy.typing import NDArray
from src.core.agent_interfaces import (AgentRuntimeConfig, EnvConfig,
                                   EvaluationHooks)
from src.core.checkpoint_io import load_checkpoint, save_checkpoint
from src.core.env_factory import build_trading_env
from src.core.schedule_builder import build_training_schedule
from src.core.visualization import TradingVisualizer
from src.metrics.metrics_calculator import MetricsCalculator
from src.core.callbacks import MetricStreamingEvalCallback
from src.metrics.metric_sink import MetricsSink, MetricsSinkConfig
from src.core.portfolio_manager import PortfolioManager
from src.core.experiments import ExperimentRegistry, hash_state_dict
from src.data.providers import DataProvider

if TYPE_CHECKING:
    from src.environment import TradingEnv

from src.core.config import DEFAULT_PPO_PARAMS, PARAM_RANGES, DEFAULT_POLICY_KWARGS
from src.utils.common import (type_check, MAX_POSITION_SIZE, MIN_POSITION_SIZE,
                          DEFAULT_STOP_LOSS, MIN_TRADE_SIZE)
import torch
from uuid import uuid4

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
                 tensorboard_log: str = "./artifacts/logs/tensorboard/",
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
        self.registry = ExperimentRegistry()
        self.provider: Optional[DataProvider] = None
        self._loaded_manifest = None
        self._expected_state_hash: Optional[str] = None
        self._force_deterministic_eval = False
        self.evaluation_metrics = {
            'returns': [],
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'information_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'win_rate': 0.0
        }

    def _get_data_index(self, stock_names: List[str], start_date: datetime,
                        end_date: datetime, provider: Optional[DataProvider]):
        try:
            if provider is None:
                raise ValueError("A data provider is required to fetch data index")
            data = fetch_trading_data(stock_names, start_date, end_date, provider)
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
                      feature_config: Optional[Dict[str, Any]] = None,
                      feature_pipeline: Optional[Any] = None,
                      training_mode: bool = True,
                      provider: Optional[DataProvider] = None) -> None:
        """Initialize trading environment with parameters.

        Args:
            stock_names: List of stock symbols
            start_date: Training start date
            end_date: Training end date
            env_params: Environment parameters (balance, costs, etc.)
            feature_config: Optional feature engineering configuration
            feature_pipeline: Optional feature processor instance or precomputed feature data
            training_mode: Whether the environment should enable training-specific behavior
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
            logger.warning("feature_config support is deprecated; please provide feature_pipeline instead.")
        if feature_pipeline is not None:
            logger.info("Using explicit feature_pipeline provided by caller.")

        if provider is None:
            raise ValueError("initialize_env requires an explicit data provider")

        try:
            self.provider = provider
            env_config = EnvConfig(
                stock_names=stock_names,
                start_date=start_date,
                end_date=end_date,
                params=cleaned_params,
                feature_config=feature_config,
                feature_pipeline=feature_pipeline,
                training_mode=training_mode,
                provider=provider,
            )
            # Initializes the TradingEnv which uses PortfolioManager with balance check in execute_trade
            self.env = build_trading_env(env_config)
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
                            feature_config: Optional[Dict[str, Any]] = None,
                            feature_pipeline: Optional[Any] = None,
                            training_mode: bool = True,
                            provider: Optional[DataProvider] = None):
        """Prepare environment and configure PPO model."""
        self.initialize_env(stock_names, start_date, end_date, env_params, feature_config, feature_pipeline, training_mode, provider)
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
                               feature_config: Optional[Dict[str, Any]],
                               feature_pipeline: Optional[Any],
                               provider: DataProvider) -> TradingEnv:
        config = EnvConfig(
            stock_names=stock_names,
            start_date=date_range[0],
            end_date=date_range[1],
            params=env_params,
            feature_config=feature_config,
            feature_pipeline=feature_pipeline,
            training_mode=False,
            provider=provider,
        )
        return build_trading_env(config)

    def _prepare_evaluation_config(self, evaluation_config: Optional[Dict[str, Any]], reports_dir: str) -> Dict[str, Any]:
        """Merge caller-supplied evaluation options with defaults and materialize a sink."""
        config: Dict[str, Any] = {
            'eval_freq': 1000,
            'seeds': [0],
            'log_dir': os.path.join(reports_dir, "eval_stream"),
            'sink': None,
            'sink_config': None,
        }

        if evaluation_config:
            config.update({k: v for k, v in evaluation_config.items() if v is not None})

        sink = config.get('sink')
        sink_config = config.get('sink_config')
        if sink is None:
            if sink_config is None:
                sink = MetricsSink()
            elif isinstance(sink_config, MetricsSinkConfig):
                sink = MetricsSink(sink_config)
            elif isinstance(sink_config, dict):
                sink = MetricsSink(MetricsSinkConfig(**sink_config))
            else:
                raise TypeError("sink_config must be a MetricsSinkConfig instance or mapping of initialization args")

        config['sink'] = sink
        return config
    
    def _calculate_training_metrics(self) -> Dict[str, float]:
        if not self.env or not self.env.portfolio_manager:
            return {}
        metrics = self.env.portfolio_manager.get_portfolio_metrics()
        # Make sure volatility is float
        metrics['volatility'] = float(metrics.get('volatility', 0.0))
        return metrics

    def _hash_current_state(self) -> str:
        if not self.model:
            raise ValueError("Model not initialized")
        return hash_state_dict(self.model.policy.state_dict())

    def _set_deterministic_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            logger.warning("Deterministic algorithms not fully available; proceeding with seeded randomness only.")
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    @type_check
    def train(self,
              stock_names: List,
              start_date: datetime,
              end_date: datetime,
              env_params: Dict[str, Any],
              ppo_params: Dict[str, Any],
              callback: Optional[BaseCallback] = None,
              feature_config: Optional[Dict[str, Any]] = None,
              feature_pipeline: Optional[Any] = None,
              checkpoint_dir: str = "artifacts/checkpoints",
              checkpoint_interval: Optional[int] = None,
              manifest_filename: str = "manifest.json",
              deterministic_eval: bool = True,
              provider: Optional[DataProvider] = None,
              evaluation_config: Optional[Dict[str, Any]] = None,
              schedule_config: Optional[Dict[str, Any]] = None,
              evaluation_hooks: Optional[EvaluationHooks] = None) -> Dict[str, float]:
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
            evaluation_config: Evaluation callback configuration block. Example:
                {
                    "eval_freq": 750,
                    "seeds": [0, 42],
                    "sink": MetricsSink(MetricsSinkConfig(jsonl_path="metrics/stream.jsonl")),
                }
            The agent composes a MetricStreamingEvalCallback from this block without
            mutating the callback itself.

        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Starting training with {len(stock_names)} stocks from {start_date} to {end_date}")
        self._force_deterministic_eval = deterministic_eval
        provider_to_use = provider or self.provider
        if provider_to_use is None:
            raise ValueError("train requires an explicit data provider")

        self._init_env_and_model(stock_names, start_date, end_date, env_params, ppo_params, feature_config, feature_pipeline, True, provider_to_use)

        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        run_dir = os.path.join(checkpoint_dir, run_id)
        reports_dir = os.path.join("artifacts", "reports", run_id)
        logs_dir = os.path.join("artifacts", "logs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        checkpoint_path = os.path.join(run_dir, "checkpoints")
        schedule = build_training_schedule(len(self.env.data), schedule_config)
        eval_config = self._prepare_evaluation_config(evaluation_config, reports_dir)
        eval_env = self._create_validation_env(stock_names, (start_date, end_date), env_params, feature_config, provider_to_use)
        eval_callback = EvalCallback(self.env,
                                   best_model_save_path=os.path.join(run_dir, "best_model"),
                                   log_path=os.path.join(logs_dir, "eval_logs"),
                                   eval_freq=1000,
                                   deterministic=True,
                                   render=False)

        metric_streaming_callback = MetricStreamingEvalCallback(
            eval_env=eval_env,
            metrics_sink=eval_config['sink'],
            visualizer=TradingVisualizer(),
            eval_freq=eval_config['eval_freq'],
            seeds=eval_config['seeds'],
            log_dir=eval_config['log_dir'],
        )

        injected_hooks = (evaluation_hooks or EvaluationHooks()).build(self.env)
        callbacks: List[BaseCallback] = [eval_callback, metric_streaming_callback, *injected_hooks]
        if checkpoint_interval and checkpoint_interval > 0:
            os.makedirs(checkpoint_path, exist_ok=True)
            callbacks.append(
                CheckpointCallback(
                    save_freq=checkpoint_interval,
                    save_path=checkpoint_path,
                    name_prefix="ppo_checkpoint",
                ))
        if callback:
            callbacks.append(callback)

        combined_callback: Optional[BaseCallback] = None
        if callbacks:
            combined_callback = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)

        try:
            self.model.learn(total_timesteps=schedule.total_timesteps,
                           callback=combined_callback)
            final_model_path = os.path.join(run_dir, "trained_model.zip")
            self.save(final_model_path)
            state_hash = self._hash_current_state()
            manifest_path = os.path.join(run_dir, manifest_filename)
            manifest = ExperimentRegistry.build_manifest(
                run_id=run_id,
                env_params=env_params,
                ppo_params=self.ppo_params,
                feature_config=feature_config,
                data_range={
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                },
                model_path=final_model_path,
                state_dict_hash=state_hash,
                artifacts=[run_dir],
                deterministic_eval=deterministic_eval)
            ExperimentRegistry.save_manifest(manifest, manifest_path)
            self.registry.register_run(manifest)
            self._expected_state_hash = state_hash
            self._loaded_manifest = manifest
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

        if self._expected_state_hash:
            actual_hash = self._hash_current_state()
            if actual_hash != self._expected_state_hash:
                raise ValueError(
                    "Loaded model state hash does not match manifest. "
                    "Reload the checkpoint to ensure determinism.")

        final_deterministic = True if self._force_deterministic_eval else deterministic
        action, _ = self.model.predict(observation, deterministic=final_deterministic)
        action = np.array(action, dtype=np.float32)
        logger.info(f"Generated action: {action}")
        return action

    @type_check
    def test(self, stock_names: List, start_date: datetime, end_date: datetime,
             env_params: Dict[str, Any], feature_config: Optional[Dict[str, Any]] = None,
             feature_pipeline: Optional[Any] = None,
             provider: Optional[DataProvider] = None) -> Dict[str, Any]:
        logger.info(f"\n{'='*50}\nStarting test run\n{'='*50}")
        logger.info(f"Stocks: {stock_names}")
        logger.info(f"Period: {start_date} to {end_date}")

        provider_to_use = provider or self.provider
        if provider_to_use is None:
            raise ValueError("test requires an explicit data provider")

        self.initialize_env(stock_names, start_date, end_date, env_params, feature_config,
                            feature_pipeline, training_mode=True, provider=provider_to_use)
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
        save_checkpoint(self.model, path)

    @type_check
    def load(self, path: str, manifest_path: Optional[str] = None,
             enforce_hash: bool = True, deterministic_eval: bool = True) -> None:
        """Load model with validation and manifest verification."""
        if not path.strip():
            raise ValueError("Empty path")
        if not self.env:
            raise ValueError("Environment not initialized")

        if enforce_hash and not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found at {path}")

        resolved_manifest_path = manifest_path or os.path.join(os.path.dirname(path), "manifest.json")
        manifest = None
        if resolved_manifest_path and os.path.exists(resolved_manifest_path):
            manifest = ExperimentRegistry.load_manifest(resolved_manifest_path)
        elif enforce_hash:
            raise FileNotFoundError(f"Manifest not found at {resolved_manifest_path}; cannot verify checkpoint integrity.")

        self.model, manifest = load_checkpoint(path, env=self.env, manifest_path=manifest_path, enforce_hash=enforce_hash)
        if enforce_hash:
            if manifest is None or not manifest.state_dict_hash:
                raise ValueError("Manifest missing state_dict_hash; cannot verify checkpoint integrity.")
            if manifest.model_path and os.path.abspath(manifest.model_path) != os.path.abspath(path):
                raise ValueError("Manifest model_path does not match the requested checkpoint path.")
            loaded_hash = self._hash_current_state()
            if loaded_hash != manifest.state_dict_hash:
                raise ValueError("Checkpoint hash mismatch between manifest and loaded state.")
            self._expected_state_hash = manifest.state_dict_hash
        else:
            self._expected_state_hash = None

        self._loaded_manifest = manifest
        self._force_deterministic_eval = manifest.deterministic_eval if manifest else deterministic_eval

    @type_check
    def load_for_inference(self,
                           model_path: str,
                           stock_names: List,
                           start_date: datetime,
                           end_date: datetime,
                           env_params: Optional[Dict[str, Any]],
                           feature_config: Optional[Dict[str, Any]] = None,
                           feature_pipeline: Optional[Any] = None,
                           manifest_path: Optional[str] = None,
                           seed: int = 42) -> None:
        """Set up an evaluation-only environment and load a model deterministically."""
        manifest = None
        if manifest_path:
            manifest = ExperimentRegistry.load_manifest(manifest_path)
            if manifest.model_path and os.path.abspath(manifest.model_path) != os.path.abspath(model_path):
                raise ValueError("Manifest model_path does not match the provided model_path.")

            feature_config = feature_config or manifest.feature_config
            env_params = env_params or manifest.env_params
            if manifest.data_range:
                start_date = datetime.fromisoformat(manifest.data_range.get('start_date')) if 'start_date' in manifest.data_range else start_date
                end_date = datetime.fromisoformat(manifest.data_range.get('end_date')) if 'end_date' in manifest.data_range else end_date
            ppo_params = manifest.ppo_params or self.ppo_params
            deterministic_eval = manifest.deterministic_eval
        else:
            ppo_params = self.ppo_params
            deterministic_eval = True

        if env_params is None:
            raise ValueError("Environment parameters must be provided directly or via the manifest for inference.")

        self._set_deterministic_seed(seed)
        self._init_env_and_model(stock_names, start_date, end_date, env_params, ppo_params,
                                 feature_config, feature_pipeline, training_mode=False)
        self.load(model_path, manifest_path=manifest_path, deterministic_eval=deterministic_eval)
        if self.env:
            self.env.reset(seed=seed)
