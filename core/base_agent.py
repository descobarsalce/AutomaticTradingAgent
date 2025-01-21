
#!/usr/bin/env python
import logging
import numpy as np
from datetime import datetime, timedelta
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from typing import Dict, Any, Optional, List, Union, Tuple, cast
from numpy.typing import NDArray

from metrics.metrics_calculator import MetricsCalculator
from environment import SimpleTradingEnv
from data.data_handler import DataHandler
from utils.common import (
    type_check,
    MAX_POSITION_SIZE,
    MIN_POSITION_SIZE,
    DEFAULT_STOP_LOSS,
    MIN_TRADE_SIZE
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PPOAgentModel:
    """Consolidated agent model for training and testing PPO-based trading strategies."""
    
    def __init__(self):
        self.agent = None
        self.env = None
        self.data_handler = DataHandler()
        self.portfolio_history = []

    def initialize_env(self, data, env_params: Dict[str, Any]):
        env_params['min_transaction_size'] = MIN_TRADE_SIZE
        self.env = SimpleTradingEnv(
            data=data,
            initial_balance=env_params['initial_balance'],
            transaction_cost=env_params['transaction_cost'],
            use_position_profit=env_params.get('use_position_profit', False),
            use_holding_bonus=env_params.get('use_holding_bonus', False),
            use_trading_penalty=env_params.get('use_trading_penalty', False),
            training_mode=True
        )

    def prepare_processed_data(self, stock_name: str, start_date: datetime, end_date: datetime):
        portfolio_data = self.data_handler.fetch_data(symbols=[stock_name], start_date=start_date, end_date=end_date)
        if not portfolio_data:
            raise ValueError("No data found in database")
        prepared_data = self.data_handler.prepare_data()
        return next(iter(prepared_data.values()))

    def train(self, stock_name: str, start_date: datetime, end_date: datetime,
              env_params: Dict[str, Any], ppo_params: Dict[str, Any],
              callback=None) -> Dict[str, float]:
        data = self.prepare_processed_data(stock_name, start_date, end_date)
        self.initialize_env(data, env_params)
        
        initial_learning_rate = ppo_params.get('learning_rate', 3e-4)
        agent_params = ppo_params.copy()
        agent_params['learning_rate'] = initial_learning_rate
        
        self.agent = TradingAgent(env=self.env, ppo_params=ppo_params)

        total_timesteps = (end_date - start_date).days
        self.agent.train(total_timesteps=total_timesteps, callback=callback)
        self.agent.save("trained_model.zip")

        self.portfolio_history = self.env.get_portfolio_history()
        if len(self.portfolio_history) > 1:
            returns = MetricsCalculator.calculate_returns(self.portfolio_history)
            return {
                'sharpe_ratio': MetricsCalculator.calculate_sharpe_ratio(returns),
                'max_drawdown': MetricsCalculator.calculate_maximum_drawdown(self.portfolio_history),
                'sortino_ratio': MetricsCalculator.calculate_sortino_ratio(returns),
                'volatility': MetricsCalculator.calculate_volatility(returns),
                'total_return': (self.portfolio_history[-1] - self.portfolio_history[0]) / self.portfolio_history[0],
                'final_value': self.portfolio_history[-1]
            }
        return {}

    def test(self, stock_name: str, start_date: datetime, end_date: datetime,
             env_params: Dict[str, Any], ppo_params: Dict[str, Any]) -> Dict[str, Any]:
        data = self.prepare_processed_data(stock_name, start_date, end_date)
        self.initialize_env(data, env_params)
        self.agent = TradingAgent(env=self.env, ppo_params=ppo_params)
        self.agent.load("trained_model.zip")

        obs, _ = self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        info_history = []

        while not done:
            action = self.agent.predict(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            info_history.append(info)

        portfolio_history = self.env.get_portfolio_history()
        returns = MetricsCalculator.calculate_returns(portfolio_history)

        from core.visualization import plot_discrete_actions, plot_actions_with_price
        action_fig = plot_discrete_actions(info_history)
        combined_fig = plot_actions_with_price(info_history, data)
        
        return {
            'portfolio_history': portfolio_history,
            'returns': returns,
            'info_history': info_history,
            'action_plot': action_fig,
            'combined_plot': combined_fig,
            'metrics': {
                'sharpe_ratio': MetricsCalculator.calculate_sharpe_ratio(returns),
                'sortino_ratio': MetricsCalculator.calculate_sortino_ratio(returns),
                'information_ratio': MetricsCalculator.calculate_information_ratio(returns),
                'max_drawdown': MetricsCalculator.calculate_maximum_drawdown(portfolio_history),
                'volatility': MetricsCalculator.calculate_volatility(returns),
                'beta': MetricsCalculator.calculate_beta(returns, data['Close'].pct_change().values)
            }
        }


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
    def __init__(self, env: Env, ppo_params: Optional[Dict[str, Union[float, int, bool, None]]] = None,
                policy_type: str = "MlpPolicy", policy_kwargs: Optional[Dict[str, Any]] = None,
                tensorboard_log: str = "./tensorboard_logs/", seed: Optional[int] = None,
                optimize_for_sharpe: bool = True) -> None:
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
            if optimize_for_sharpe:
                policy_kwargs = {'net_arch': [dict(pi=[128, 128, 128], vf=[128, 128, 128])]}
            else:
                policy_kwargs = self.DEFAULT_POLICY_KWARGS.copy()

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
        return self.evaluation_metrics.copy()

    @type_check
    def save(self, path: str) -> None:
        if not path.strip():
            raise ValueError("path cannot be empty")
        self.model.save(path)

    @type_check
    def load(self, path: str) -> None:
        if not path.strip():
            raise ValueError("path cannot be empty")
        self.model = PPO.load(path)


class TradingAgent(BaseAgent):
    """Trading agent implementation using PPO algorithm for discrete action trading."""

    def __init__(self, env: Env, ppo_params: Optional[Dict[str, Union[float, int, bool, None]]] = None,
                seed: Optional[int] = None) -> None:
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
        action = super().predict(observation, deterministic)

        if not isinstance(action, (np.ndarray, int)):
            raise ValueError("Invalid action type")

        action = int(action)
        if not 0 <= action <= 2:
            raise ValueError(f"Invalid action value: {action}")

        return np.array([action])

    def update_state(self, portfolio_value: float, positions: Dict[str, float]) -> None:
        for symbol, size in positions.items():
            if not isinstance(size, (int, float)) or not np.isfinite(size):
                raise ValueError(f"Invalid position size type for {symbol}")
            if abs(size) > self.max_position_size:
                raise ValueError(f"Position size {size} exceeds limit for {symbol}")

        super().update_state(portfolio_value, positions)

    def train(self, total_timesteps: int, callback: Optional[BaseCallback] = None) -> None:
        if not isinstance(total_timesteps, int) or total_timesteps <= 0:
            raise ValueError("total_timesteps must be a positive integer")

        import time
        self.start_time = time.time()

        if callback:
            self.model.learn(total_timesteps=total_timesteps, callback=callback)
        else:
            self.model.learn(total_timesteps=total_timesteps)
