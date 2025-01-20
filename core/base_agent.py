
"""Base agent implementation with PPO functionality"""

from typing import Dict, Any, Optional, List, Union, Tuple, cast
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import numpy as np
from numpy.typing import NDArray
from utils.common import type_check, MAX_POSITION_SIZE, MIN_POSITION_SIZE, DEFAULT_STOP_LOSS
from metrics.metrics_calculator import MetricsCalculator
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class BaseAgent:
    """Base agent class implementing core PPO functionality."""

    DEFAULT_PPO_PARAMS = {
        'learning_rate': 3e-4,
        'n_steps': 1024,
        'batch_size': 256,
        'n_epochs': 25,
        'gamma': 0.99,
        'gae_lambda': 0.98,
        'clip_range': 0.0,
        'ent_coef': 0.01,
        'vf_coef': 0.8,
        'max_grad_norm': 0.3,
        'use_sde': False,
        'target_kl': 0.05,
        'verbose': 0
    }

    PARAM_RANGES = {
        'learning_rate': (1e-5, 5e-4),
        'n_steps': (512, 2048),
        'batch_size': (64, 512),
        'n_epochs': (3, 10),
        'gamma': (0.95, 0.999),
        'gae_lambda': (0.9, 0.99),
        'clip_range': (0.0, 0.3),
        'ent_coef': (0.001, 0.02),
        'vf_coef': (0.4, 0.9),
        'max_grad_norm': (0.3, 0.8),
        'target_kl': (0.02, 0.1)
    }

    DEFAULT_POLICY_KWARGS = {
        'net_arch': [dict(pi=[64, 64], vf=[64, 64])]
    }

    @type_check
    def __init__(self, env: Env,
                ppo_params: Optional[Dict[str, Union[float, int, bool, None]]] = None,
                policy_type: str = "MlpPolicy",
                policy_kwargs: Optional[Dict[str, Any]] = None,
                tensorboard_log: str = "./tensorboard_logs/",
                seed: Optional[int] = None,
                optimize_for_sharpe: bool = True) -> None:
        """Initialize base agent with PPO configuration."""
        if not isinstance(env, Env):
            raise TypeError("env must be a valid Gymnasium environment")
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValueError("seed must be a positive integer if provided")

        self.env = env
        self.initial_balance = getattr(env, 'initial_balance', 100000.0)
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

        if ppo_params is None:
            ppo_params = self.DEFAULT_PPO_PARAMS.copy()
        else:
            for param, value in ppo_params.items():
                if param in self.PARAM_RANGES and value is not None:
                    min_val, max_val = self.PARAM_RANGES[param]
                    if not (min_val <= value <= max_val):
                        logger.warning(f"Parameter {param} value {value} outside recommended range [{min_val}, {max_val}]")
                        ppo_params[param] = max(min_val, min(value, max_val))

        if policy_kwargs is None:
            policy_kwargs = {
                'net_arch': [dict(pi=[128, 128, 128], vf=[128, 128, 128])]
            } if optimize_for_sharpe else self.DEFAULT_POLICY_KWARGS.copy()

        try:
            if 'verbose' not in ppo_params:
                ppo_params['verbose'] = 1

            self.model = PPO(
                policy_type,
                env,
                **ppo_params,
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
                seed=seed
            )
        except Exception as e:
            logger.exception("Failed to initialize PPO model")
            raise

    @type_check
    def train(self, total_timesteps: int) -> None:
        """Train the agent for specified timesteps."""
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

            self.model.learn(
                total_timesteps=total_timesteps,
                callback=eval_callback
            )
        except Exception as e:
            logger.exception("Error during training")
            raise

    @type_check
    def predict(self, observation: NDArray, deterministic: bool = True) -> NDArray:
        """Make a prediction based on current observation."""
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
        """Update agent's state with current portfolio information."""
        if not isinstance(portfolio_value, (int, float)) or portfolio_value < 0:
            raise ValueError("Invalid portfolio value")
        if not isinstance(positions, dict):
            raise TypeError("positions must be a dictionary")

        try:
            current_prices = {symbol: self.env.data.loc[self.env.current_step, 'Close'] for symbol in positions}
            calculated_positions_value = sum(size * current_prices[symbol] for symbol, size in positions.items())
            cash_balance = getattr(self.env, 'balance', 0.0)
            calculated_portfolio = calculated_positions_value + cash_balance

            if not np.isclose(calculated_portfolio, portfolio_value, rtol=1e-3):
                logger.warning(f"Portfolio value inconsistency: calculated={calculated_portfolio:.2f}, reported={portfolio_value:.2f}")
        except Exception as e:
            logger.error(f"Error during portfolio check: {str(e)}")

        self.portfolio_history.append(portfolio_value)
        self.positions_history.append(positions)
        self._update_metrics()

    def _update_metrics(self) -> None:
        """Update performance metrics."""
        try:
            metrics_calc = MetricsCalculator()
            returns = metrics_calc.calculate_returns(self.portfolio_history)

            self.evaluation_metrics.update({
                'returns': float(np.mean(returns)) if len(returns) > 0 else 0.0,
                'sharpe_ratio': metrics_calc.calculate_sharpe_ratio(returns),
                'sortino_ratio': metrics_calc.calculate_sortino_ratio(returns),
                'information_ratio': metrics_calc.calculate_information_ratio(returns),
                'max_drawdown': metrics_calc.calculate_maximum_drawdown(self.portfolio_history)
            })

            valid_positions = [p for p in self.positions_history if isinstance(p, dict)]
            self.evaluation_metrics['total_trades'] = len(valid_positions)

            if valid_positions:
                profitable_trades = sum(1 for i in range(1, len(valid_positions))
                                   if sum(valid_positions[i].values()) > sum(valid_positions[i-1].values()))
                self.evaluation_metrics['win_rate'] = profitable_trades / len(valid_positions)

        except Exception as e:
            logger.exception("Error updating metrics")
            self.evaluation_metrics = {k: 0.0 for k in self.evaluation_metrics}

    @type_check
    def get_metrics(self) -> Dict[str, Union[float, List[float], int]]:
        """Get current evaluation metrics."""
        return self.evaluation_metrics.copy()

    @type_check
    def save(self, path: str) -> None:
        """Save model to path."""
        if not path.strip():
            raise ValueError("path cannot be empty")
        self.model.save(path)

    @type_check
    def load(self, path: str) -> None:
        """Load model from path."""
        if not path.strip():
            raise ValueError("path cannot be empty")
        self.model = PPO.load(path)


class TradingAgent(BaseAgent):
    """Trading agent implementation using PPO algorithm for discrete action trading."""

    def __init__(self, env: Env,
                ppo_params: Optional[Dict[str, Union[float, int, bool, None]]] = None,
                seed: Optional[int] = None) -> None:
        """Initialize trading agent with discrete actions."""
        if ppo_params and 'learning_rate' in ppo_params:
            if not isinstance(ppo_params['learning_rate'], (int, float)):
                raise TypeError("learning_rate must be a number")
        
        super().__init__(
            env,
            ppo_params=ppo_params,
            seed=seed,
            policy_kwargs={
                'net_arch': dict(pi=[64, 64], vf=[64, 64])
            })

        self.max_position_size = MAX_POSITION_SIZE
        self.min_position_size = MIN_POSITION_SIZE
        self.stop_loss = DEFAULT_STOP_LOSS

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Generate trading decisions (0=hold, 1=buy, 2=sell)."""
        action = super().predict(observation, deterministic)

        if not isinstance(action, (np.ndarray, int)):
            raise ValueError("Invalid action type")

        action = int(action)
        if not 0 <= action <= 2:
            raise ValueError(f"Invalid action value: {action}")

        return np.array([action])

    def update_state(self, portfolio_value: float, positions: Dict[str, float]) -> None:
        """Update agent's state with current portfolio information."""
        for symbol, size in positions.items():
            if not isinstance(size, (int, float)) or not np.isfinite(size):
                raise ValueError(f"Invalid position size type for {symbol}")
            if abs(size) > self.max_position_size:
                raise ValueError(f"Position size {size} exceeds limit for {symbol}")

        super().update_state(portfolio_value, positions)

    def train(self, total_timesteps: int, callback: Optional[BaseCallback] = None) -> None:
        """Train the agent on historical market data."""
        if not isinstance(total_timesteps, int) or total_timesteps <= 0:
            raise ValueError("total_timesteps must be a positive integer")

        import time
        self.start_time = time.time()

        if callback:
            self.model.learn(total_timesteps=total_timesteps, callback=callback)
        else:
            self.model.learn(total_timesteps=total_timesteps)
