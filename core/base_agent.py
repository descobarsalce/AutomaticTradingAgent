from typing import Dict, Any, Optional, List, Union, Tuple
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from utils.callbacks import ProgressBarCallback
import numpy as np
from numpy.typing import NDArray
from utils.decorators import type_check
from .metrics import MetricsCalculator
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class BaseAgent:
    # Default PPO parameters optimized for financial trading
    DEFAULT_PPO_PARAMS: Dict[str, Union[float, int, bool, None]] = {
        'learning_rate': 5e-2,
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
        self.initial_balance = getattr(env, 'initial_balance', 100000.0)  # Default to 100k if not specified
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
            # Set default verbose level if not specified in ppo_params
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
        """
        Update agent's state tracking with new portfolio information.
        
        Args:
            portfolio_value: Current portfolio value
            positions: Dictionary of symbol to position size mappings
            
        Raises:
            TypeError: If input types are invalid
            ValueError: If values are invalid or inconsistent
            
        This method implements comprehensive state tracking and validation including:
        - Portfolio value and position size validation
        - Maximum drawdown monitoring
        - Total portfolio exposure checks
        - Trade frequency tracking
        - Market impact consideration
        """
        # Basic type validation
        if not isinstance(portfolio_value, (int, float)):
            raise TypeError("portfolio_value must be a number")
        if portfolio_value < 0:
            raise ValueError("portfolio_value cannot be negative")
        if not isinstance(positions, dict):
            raise TypeError("positions must be a dictionary")
        if not all(isinstance(v, (int, float)) for v in positions.values()):
            raise TypeError("All position values must be numbers")
            
        # Validate position sizes
        for symbol, size in positions.items():
            if not np.isfinite(size):
                raise ValueError(f"Position size for {symbol} must be finite")
            if abs(size) > 10.0:  # Maximum allowed position size
                logger.warning(f"Large position detected for {symbol}: {size}")
        
        # Portfolio consistency check
        try:
            current_prices = {
                symbol: self.env.data.loc[self.env.current_step, 'Close'] 
                for symbol in positions.keys()
            }
            calculated_positions_value = sum(
                size * current_prices[symbol] 
                for symbol, size in positions.items()
            )
            
            # Get current cash balance from environment
            cash_balance = getattr(self.env, 'balance', 0.0)
            calculated_portfolio = calculated_positions_value + cash_balance
            
            # Allow for small numerical differences
            if not np.isclose(calculated_portfolio, portfolio_value, rtol=1e-3):
                logger.warning(
                    f"Portfolio value inconsistency detected: "
                    f"calculated={calculated_portfolio:.2f}, "
                    f"reported={portfolio_value:.2f}, "
                    f"difference={abs(calculated_portfolio - portfolio_value):.2f}"
                )
        except Exception as e:
            logger.error(f"Error during portfolio consistency check: {str(e)}")
            
        # Track state changes and validate portfolio health
        previous_portfolio = self.portfolio_history[-1] if self.portfolio_history else self.initial_balance
        previous_positions = self.positions_history[-1] if self.positions_history else {}
        
        # Calculate and validate maximum drawdown
        max_portfolio_value = max(self.portfolio_history) if self.portfolio_history else previous_portfolio
        current_drawdown = (max_portfolio_value - portfolio_value) / max_portfolio_value
        if current_drawdown > 0.20:  # 20% max drawdown limit
            logger.warning(f"Maximum drawdown limit exceeded: {current_drawdown:.2%}")
        
        # Calculate and validate total portfolio exposure
        total_exposure = sum(abs(size) for size in positions.values())
        if total_exposure > 2.0:  # Maximum 200% total exposure
            logger.warning(f"Total portfolio exposure exceeds limit: {total_exposure:.2f}x")
        
        # Track trade frequency
        current_time = getattr(self.env, 'current_step', 0)
        if not hasattr(self, 'last_trade_time'):
            self.last_trade_time = {}
        
        # Monitor position changes and market impact
        if previous_positions is not None:
            for symbol, current_size in positions.items():
                prev_size = previous_positions.get(symbol, 0.0)
                size_change = abs(current_size - prev_size)
                
                if size_change > 0.1:  # Significant position change
                    # Track trade timing
                    last_trade = self.last_trade_time.get(symbol, 0)
                    time_since_last_trade = current_time - last_trade
                    
                    if time_since_last_trade < 5:  # Minimum 5 steps between trades
                        logger.warning(f"High frequency trading detected for {symbol}: {time_since_last_trade} steps since last trade")
                    
                    # Update last trade time
                    self.last_trade_time[symbol] = current_time
                    
                    # Log position change
                    logger.info(f"Position change for {symbol}: {prev_size:.2f} -> {current_size:.2f}")
                    
                    # Market impact warning for large trades
                    try:
                        current_volume = self.env.data.iloc[current_time]['Volume']
                        if size_change * self.env.data.iloc[current_time]['Close'] > 0.1 * current_volume:
                            logger.warning(f"Large market impact potential for {symbol}: trade size > 10% of volume")
                    except (AttributeError, KeyError, IndexError):
                        pass  # Skip volume check if data not available
        
        # Calculate and log portfolio metrics
        pct_change = ((portfolio_value - previous_portfolio) / previous_portfolio) * 100
        if abs(pct_change) > 1.0:  # Log changes greater than 1%
            logger.info(f"Significant portfolio value change: {pct_change:.2f}%")
            
            # Additional volatility check
            if len(self.portfolio_history) >= 10:
                returns = np.diff(self.portfolio_history[-10:]) / self.portfolio_history[-11:-1]
                volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
                if volatility > 0.4:  # 40% annualized volatility threshold
                    logger.warning(f"High portfolio volatility detected: {volatility:.2%}")
        
        # Update state
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
