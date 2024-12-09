from typing import Dict, Any, Optional, List, Union, Tuple
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from callbacks import ProgressBarCallback
import numpy as np
from numpy.typing import NDArray
from decorators import type_check
from .metrics import MetricsCalculator
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class BaseAgent:
    # Default PPO parameters optimized for financial trading
    DEFAULT_PPO_PARAMS: Dict[str, Union[float, int, bool, None]] = {
        'learning_rate': 5e-3,
        'n_steps': 256,
        'batch_size': 256,
        'n_epochs': 5,
        'gamma': 0.99,
        'gae_lambda': 0.98,
        'clip_range': 0.1,
        'ent_coef': 0.01,
        'vf_coef': 0.8,
        'max_grad_norm': 0.3,
        'use_sde': True,
        'sde_sample_freq': 4,
        'target_kl': 0.015
    }

    # Parameter ranges for optimization
    PARAM_RANGES = {
        'learning_rate': (5e-5, 5e-4),
        'n_steps': (512, 2048),
        'batch_size': (64, 256),
        'n_epochs': (3, 10),
        'gamma': (0.95, 0.999),
        'gae_lambda': (0.9, 0.99),
        'clip_range': (0.1, 0.3),
        'ent_coef': (0.001, 0.02),
        'vf_coef': (0.4, 0.9),
        'max_grad_norm': (0.3, 0.8),
        'target_kl': (0.01, 0.03)
    }

    # Default policy network parameters
    DEFAULT_POLICY_KWARGS: Dict[str, Any] = {
        'net_arch': [dict(pi=[64, 64], vf=[64, 64])]
    }

    @type_check
    def __init__(
        self,
        env: Env,
        ppo_params: Optional[Dict[str, Union[float, int, bool, None]]] = None,
        policy_type: str = "MlpPolicy",
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: str = "./tensorboard_logs/",
        seed: Optional[int] = None,
        optimize_for_sharpe: bool = True
    ) -> None:
        """Type-safe initialization of the base agent."""
        """
        Initialize the base agent with advanced configuration and state tracking.
        """
        if not isinstance(env, Env):
            raise TypeError("env must be a valid Gymnasium environment")
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValueError("seed must be a positive integer if provided")
        if not isinstance(tensorboard_log, str):
            raise TypeError("tensorboard_log must be a string")
            
        # Store environment
        self.env = env
            
        # Initialize instance variables
        self.portfolio_history = []
        self.positions_history = []
        self.evaluation_metrics = {
            'returns': [],
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'information_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'win_rate': 0.0
        }

        # Set up and validate PPO parameters
        if ppo_params is None:
            ppo_params = self.DEFAULT_PPO_PARAMS.copy()
        else:
            # Validate parameter ranges
            for param, value in ppo_params.items():
                if param in self.PARAM_RANGES and value is not None:
                    min_val, max_val = self.PARAM_RANGES[param]
                    if not (min_val <= value <= max_val):
                        logger.warning(f"Parameter {param} value {value} outside recommended range [{min_val}, {max_val}]")
                        ppo_params[param] = max(min_val, min(value, max_val))
        
        # Set up policy network parameters
        if policy_kwargs is None:
            if optimize_for_sharpe:
                policy_kwargs = {
                    'net_arch': [dict(pi=[128, 128, 128], vf=[128, 128, 128])]
                }
            else:
                policy_kwargs = self.DEFAULT_POLICY_KWARGS.copy()

        try:
            self.model = PPO(
                policy_type,
                env,
                **ppo_params,
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
                seed=seed,
                verbose=1
            )
        except Exception as e:
            logger.exception("Failed to initialize PPO model")
            raise

    @type_check
    def train(self, total_timesteps: int) -> None:
        """Train the agent for a specified number of timesteps."""
        if not isinstance(total_timesteps, int) or total_timesteps <= 0:
            raise ValueError("total_timesteps must be a positive integer")
        try:
            eval_callback = EvalCallback(
                self.model.get_env(),
                best_model_save_path='./best_model/',
                log_path='./eval_logs/',
                eval_freq=1000,
                deterministic=True,
                render=False
            )
            
            progress_callback = ProgressBarCallback(total_timesteps)
            
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=[eval_callback, progress_callback]
            )
        except Exception as e:
            logger.exception("Error during training")
            raise

    @type_check
    def predict(self, observation: NDArray, deterministic: bool = True) -> NDArray:
        """Make a prediction based on the current observation."""
        if not isinstance(observation, np.ndarray):
            raise TypeError("observation must be a numpy array")
        if observation.shape != self.model.observation_space.shape:
            raise ValueError(f"observation shape {observation.shape} does not match environment shape {self.model.observation_space.shape}")
        try:
            action, _states = self.model.predict(observation, deterministic=deterministic)
            return action
        except Exception as e:
            logger.exception("Error in predict method")
            raise

    @type_check
    def update_state(self, portfolio_value: Union[int, float], positions: Dict[str, Union[int, float]]) -> None:
        """Update agent's state tracking with new portfolio information."""
        if not isinstance(portfolio_value, (int, float)):
            raise TypeError("portfolio_value must be a number")
        if portfolio_value < 0:
            raise ValueError("portfolio_value cannot be negative")
        if not isinstance(positions, dict):
            raise TypeError("positions must be a dictionary")
        if not all(isinstance(v, (int, float)) for v in positions.values()):
            raise TypeError("All position values must be numbers")
            
        self.portfolio_history.append(portfolio_value)
        self.positions_history.append(positions)
        self._update_metrics()

    def _update_metrics(self) -> None:
        """Update evaluation metrics based on current state."""
        try:
            # Use MetricsCalculator for all metrics calculations
            metrics_calc = MetricsCalculator()
            
            # Calculate returns
            returns = metrics_calc.calculate_returns(self.portfolio_history)
            if len(returns) > 0:
                # Store mean return instead of returns list for easier comparison
                self.evaluation_metrics['returns'] = float(np.mean(returns))
            else:
                self.evaluation_metrics['returns'] = 0.0
            
            # Calculate risk-adjusted metrics
            self.evaluation_metrics['sharpe_ratio'] = metrics_calc.calculate_sharpe_ratio(returns)
            self.evaluation_metrics['sortino_ratio'] = metrics_calc.calculate_sortino_ratio(returns)
            self.evaluation_metrics['information_ratio'] = metrics_calc.calculate_information_ratio(returns)
            self.evaluation_metrics['max_drawdown'] = metrics_calc.calculate_maximum_drawdown(self.portfolio_history)
            
            # Update trade statistics
            valid_positions = [p for p in self.positions_history if isinstance(p, dict)]
            self.evaluation_metrics['total_trades'] = len(valid_positions)
            
            if valid_positions:
                profitable_trades = sum(1 for i in range(1, len(valid_positions))
                                   if sum(valid_positions[i].values()) > sum(valid_positions[i-1].values()))
                self.evaluation_metrics['win_rate'] = profitable_trades / len(valid_positions)
                
        except Exception as e:
            logger.exception("Error updating metrics")
            self.evaluation_metrics.update({
                'returns': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'information_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'win_rate': 0.0
            })

    @type_check
    def get_metrics(self) -> Dict[str, Union[float, List[float], int]]:
        """Get current evaluation metrics."""
        return self.evaluation_metrics.copy()

    @type_check
    def save(self, path: str) -> None:
        """Save the model to the specified path."""
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        if not path.strip():
            raise ValueError("path cannot be empty")
        self.model.save(path)

    @type_check
    def load(self, path: str) -> None:
        """Load the model from the specified path."""
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        if not path.strip():
            raise ValueError("path cannot be empty")
        self.model = PPO.load(path)
