import numpy as np
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from callbacks import ProgressBarCallback
from numpy.typing import NDArray
from decorators import type_check

# Configure logging at the module level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add the handlers to the logger
if not logger.handlers:
    logger.addHandler(ch)

class TradingAgent:
    # Default PPO parameters
    DEFAULT_PPO_PARAMS: Dict[str, Union[float, int, bool, None]] = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_sde': False,
        'sde_sample_freq': -1,
        'target_kl': None
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
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize the trading agent with advanced configuration and state tracking.
        
        Args:
            env: Gymnasium environment for training and evaluation
            ppo_params: Optional PPO algorithm parameters with specific types
            policy_type: Type of policy network to use (must be a valid policy type)
            policy_kwargs: Optional policy network parameters
            tensorboard_log: Directory for tensorboard logs (must be a valid path)
            seed: Random seed for reproducibility (must be a positive integer if provided)
            
        Raises:
            ValueError: If invalid parameter values are provided
            TypeError: If parameters are of incorrect type
        """
        # Input validation
        if not isinstance(env, Env):
            raise TypeError("env must be a valid Gymnasium environment")
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValueError("seed must be a positive integer if provided")
        if not isinstance(tensorboard_log, str):
            raise TypeError("tensorboard_log must be a string")
            
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
        # Set up default PPO parameters if none provided
        if ppo_params is None:
            ppo_params = self.DEFAULT_PPO_PARAMS.copy()
            
        # Set up default policy network parameters if none provided
        if policy_kwargs is None:
            policy_kwargs = self.DEFAULT_POLICY_KWARGS.copy()
            
        try:
            # Initialize PPO model with advanced configuration
            self.model = PPO(
                policy_type,
                env,
                learning_rate=ppo_params['learning_rate'],
                n_steps=ppo_params['n_steps'],
                batch_size=ppo_params['batch_size'],
                n_epochs=ppo_params['n_epochs'],
                gamma=ppo_params['gamma'],
                gae_lambda=ppo_params['gae_lambda'],
                clip_range=ppo_params['clip_range'],
                clip_range_vf=None,
                ent_coef=ppo_params['ent_coef'],
                vf_coef=ppo_params['vf_coef'],
                max_grad_norm=ppo_params['max_grad_norm'],
                use_sde=ppo_params['use_sde'],
                sde_sample_freq=ppo_params['sde_sample_freq'],
                target_kl=ppo_params['target_kl'],
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
        """
        Train the agent for a specified number of timesteps.
        
        Args:
            total_timesteps: Number of timesteps to train for (must be positive)
            
        Raises:
            ValueError: If total_timesteps is not positive
            RuntimeError: If training fails
        """
        if not isinstance(total_timesteps, int) or total_timesteps <= 0:
            raise ValueError("total_timesteps must be a positive integer")
        try:
            # Set up callbacks for training monitoring
            eval_callback = EvalCallback(
                self.model.get_env(),
                best_model_save_path='./best_model/',
                log_path='./eval_logs/',
                eval_freq=1000,
                deterministic=True,
                render=False
            )
            
            progress_callback = ProgressBarCallback(total_timesteps)
            
            # Train the model with callbacks
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=[eval_callback, progress_callback]
            )
            
        except Exception as e:
            logger.exception("Error during training")
            raise
            
    @type_check
    def predict(self, observation: NDArray, deterministic: bool = True) -> NDArray:
        """
        Make a prediction based on the current observation.
        
        Args:
            observation: Current environment observation (must match environment observation space)
            deterministic: Whether to use deterministic actions
            
        Returns:
            action: Predicted action matching the environment action space
            
        Raises:
            ValueError: If observation shape doesn't match environment observation space
            TypeError: If observation is not a numpy array
        """
        if not isinstance(observation, np.ndarray):
            raise TypeError("observation must be a numpy array")
        if observation.shape != self.model.observation_space.shape:
            raise ValueError(f"observation shape {observation.shape} does not match environment shape {self.model.observation_space.shape}")
        try:
            action, _states = self.model.predict(
                observation,
                deterministic=deterministic
            )
            return action
            
        except Exception as e:
            logger.exception("Error in predict method")
            raise
            
    @type_check
    def update_state(self, portfolio_value: Union[int, float], positions: Dict[str, Union[int, float]]) -> None:
        """
        Update agent's state tracking with new portfolio information.
        
        Args:
            portfolio_value: Current portfolio value (must be non-negative)
            positions: Dictionary of current positions (symbol: weight)
            
        Raises:
            ValueError: If portfolio_value is negative or positions contain invalid values
            TypeError: If arguments are of incorrect types
        """
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
        
    def _calculate_returns(self) -> np.ndarray:
        """
        Calculate returns from portfolio history with improved error handling
        and validation.
        
        Returns:
            numpy.ndarray: Array of returns or empty array if calculation fails
        """
        if len(self.portfolio_history) <= 1:
            logger.debug("Insufficient data points for returns calculation")
            return np.array([])
            
        try:
            # Validate portfolio values
            portfolio_array = np.array(self.portfolio_history)
            if not np.all(np.isfinite(portfolio_array)):
                logger.warning("Non-finite values found in portfolio history")
                portfolio_array = portfolio_array[np.isfinite(portfolio_array)]
            
            if len(portfolio_array) <= 1:
                logger.debug("Insufficient valid data points after filtering")
                return np.array([])
                
            # Calculate returns with validation
            denominator = portfolio_array[:-1]
            if np.any(denominator == 0):
                logger.warning("Zero values found in portfolio history")
                return np.array([])
                
            returns = np.diff(portfolio_array) / denominator
            
            # Filter out extreme values
            returns = returns[np.isfinite(returns)]
            if len(returns) > 0:
                # Remove extreme outliers (beyond 5 standard deviations)
                mean, std = np.mean(returns), np.std(returns)
                returns = returns[np.abs(returns - mean) <= 5 * std]
                logger.debug(f"Calculated returns with {len(returns)} valid data points")
                
            return returns
            
        except Exception as e:
            logger.exception("Error calculating returns")
            return np.array([])

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sortino ratio from returns with error handling and validation.
        Similar to Sharpe ratio but only penalizes downside volatility.
        
        Args:
            returns: numpy.ndarray of return values
            
        Returns:
            float: Annualized Sortino ratio or 0.0 if calculation fails
        """
        if not isinstance(returns, np.ndarray):
            logger.warning("Invalid input type for Sortino ratio calculation")
            return 0.0
            
        if len(returns) <= 252:  # Minimum one year of data for meaningful ratio
            logger.debug(f"Insufficient data points for reliable Sortino ratio: {len(returns)}")
            return 0.0
            
        try:
            # Remove any non-finite values
            valid_returns = returns[np.isfinite(returns)]
            if len(valid_returns) <= 1:
                logger.debug("Insufficient valid return values for Sortino ratio calculation")
                return 0.0
                
            # Calculate average return
            avg_return = np.mean(valid_returns)
            
            # Calculate downside deviation (only negative returns)
            negative_returns = valid_returns[valid_returns < 0]
            if len(negative_returns) == 0:
                logger.debug("No negative returns found for Sortino ratio calculation")
                return float('inf') if avg_return > 0 else 0.0
                
            downside_std = np.std(negative_returns, ddof=1)
            
            # Check for numerical stability
            if not np.isfinite(avg_return) or not np.isfinite(downside_std):
                logger.warning("Non-finite values in Sortino ratio calculation")
                return 0.0
                
            # Calculate annualized Sortino ratio with validation
            if downside_std > 1e-8:  # Avoid division by very small numbers
                annualization_factor = np.sqrt(252)  # Assuming daily returns
                sortino = (avg_return / downside_std) * annualization_factor
                sortino_clipped = float(np.clip(sortino, -100, 100))  # Limit extreme values
                logger.debug(f"Calculated Sortino ratio: {sortino_clipped}")
                return sortino_clipped
            else:
                logger.warning("Downside deviation too small for reliable Sortino ratio")
                return 0.0
                
        except Exception as e:
            logger.exception("Error calculating Sortino ratio")
            return 0.0

    def _calculate_information_ratio(self, returns: np.ndarray, benchmark_returns: Optional[np.ndarray] = None) -> float:
        """
        Calculate Information ratio from returns with error handling and validation.
        Measures risk-adjusted excess returns relative to a benchmark.
        
        Args:
            returns: numpy.ndarray of return values
            benchmark_returns: Optional numpy.ndarray of benchmark return values
            
        Returns:
            float: Information ratio or 0.0 if calculation fails
        """
        if not isinstance(returns, np.ndarray):
            logger.warning("Invalid input type for Information ratio calculation")
            return 0.0
            
        if len(returns) <= 252:  # Minimum one year of data
            logger.debug(f"Insufficient data points for reliable Information ratio: {len(returns)}")
            return 0.0
            
        try:
            # If no benchmark provided, use risk-free rate of 0
            if benchmark_returns is None:
                benchmark_returns = np.zeros_like(returns)
            
            # Ensure arrays are the same length
            min_length = min(len(returns), len(benchmark_returns))
            returns = returns[:min_length]
            benchmark_returns = benchmark_returns[:min_length]
            
            # Calculate excess returns
            excess_returns = returns - benchmark_returns
            
            # Remove any non-finite values
            valid_returns = excess_returns[np.isfinite(excess_returns)]
            if len(valid_returns) <= 1:
                logger.debug("Insufficient valid return values for Information ratio calculation")
                return 0.0
                
            # Calculate average excess return and tracking error
            avg_excess_return = np.mean(valid_returns)
            tracking_error = np.std(valid_returns, ddof=1)
            
            # Check for numerical stability
            if not np.isfinite(avg_excess_return) or not np.isfinite(tracking_error):
                logger.warning("Non-finite values in Information ratio calculation")
                return 0.0
                
            # Calculate Information ratio with validation
            if tracking_error > 1e-8:  # Avoid division by very small numbers
                information_ratio = avg_excess_return / tracking_error
                information_ratio_clipped = float(np.clip(information_ratio, -100, 100))
                logger.debug(f"Calculated Information ratio: {information_ratio_clipped}")
                return information_ratio_clipped
            else:
                logger.warning("Tracking error too small for reliable Information ratio")
                return 0.0
                
        except Exception as e:
            logger.exception("Error calculating Information ratio")
            return 0.0

        """
        Calculate Sharpe ratio from returns with improved error handling
        and validation.
        
        Args:
            returns: numpy.ndarray of return values
            
        Returns:
            float: Annualized Sharpe ratio or 0.0 if calculation fails
        """
        if not isinstance(returns, np.ndarray):
            logger.warning("Invalid input type for returns calculation")
            return 0.0
            
        if len(returns) <= 252:  # Minimum one year of data for meaningful Sharpe ratio
            logger.debug(f"Insufficient data points for reliable Sharpe ratio: {len(returns)}")
            return 0.0
            
        try:
            # Remove any remaining non-finite values
            valid_returns = returns[np.isfinite(returns)]
            if len(valid_returns) <= 1:
                logger.debug("Insufficient valid return values for Sharpe ratio calculation")
                return 0.0
                
            # Calculate with improved precision
            avg_return = np.mean(valid_returns)
            std_return = np.std(valid_returns, ddof=1)  # Use unbiased estimator
            
            # Check for numerical stability
            if not np.isfinite(avg_return) or not np.isfinite(std_return):
                logger.warning("Non-finite values in Sharpe ratio calculation")
                return 0.0
                
            # Calculate annualized Sharpe ratio with validation
            if std_return > 1e-8:  # Avoid division by very small numbers
                annualization_factor = np.sqrt(252)  # Assuming daily returns
                sharpe = (avg_return / std_return) * annualization_factor
                sharpe_clipped = float(np.clip(sharpe, -100, 100))  # Limit extreme values
                logger.debug(f"Calculated Sharpe ratio: {sharpe_clipped}")
                return sharpe_clipped
            else:
                logger.warning("Standard deviation too small for reliable Sharpe ratio")
                return 0.0
                
        except Exception as e:
            logger.exception("Error calculating Sharpe ratio")
            return 0.0

    def _calculate_maximum_drawdown(self) -> float:
        """
        Calculate maximum drawdown from portfolio history with improved validation
        and error handling.
        
        Returns:
            float: Maximum drawdown value between 0.0 and 1.0
        """
        if len(self.portfolio_history) <= 1:
            logger.debug("Insufficient data points for drawdown calculation")
            return 0.0
        try:
            # Convert to numpy array for efficient calculation
            values = np.array([v for v in self.portfolio_history if isinstance(v, (int, float)) and v >= 0])
            if len(values) <= 1:
                logger.warning("No valid values for drawdown calculation after filtering")
                return 0.0
                
            # Calculate running maximum
            peak = np.maximum.accumulate(values)
            # Calculate drawdown for each point
            drawdowns = (peak - values) / peak
            # Get maximum drawdown
            max_dd = float(np.nanmax(drawdowns))
            
            logger.debug(f"Calculated maximum drawdown: {max_dd:.4f}")
            return max_dd
            
        except Exception as e:
            logger.exception("Error calculating maximum drawdown")
            return 0.0

    def _update_metrics(self) -> None:
        """Update evaluation metrics based on current state with improved error handling"""
        try:
            # Calculate and validate returns
            returns = self._calculate_returns()
            if len(returns) > 0:
                self.evaluation_metrics['returns'] = returns.tolist()
            
            # Calculate Sharpe ratio
            self.evaluation_metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
            
            # Calculate Sortino ratio
            self.evaluation_metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
            
            # Calculate Information ratio (using zero benchmark for now)
            self.evaluation_metrics['information_ratio'] = self._calculate_information_ratio(returns)
            
            # Calculate maximum drawdown
            self.evaluation_metrics['max_drawdown'] = self._calculate_maximum_drawdown()
            
            # Update trade statistics with validation
            valid_positions = [p for p in self.positions_history if isinstance(p, dict)]
            self.evaluation_metrics['total_trades'] = len(valid_positions)
            
            # Calculate win rate if there are any trades
            if valid_positions:
                profitable_trades = sum(1 for i in range(1, len(valid_positions))
                                     if sum(valid_positions[i].values()) > sum(valid_positions[i-1].values()))
                self.evaluation_metrics['win_rate'] = profitable_trades / len(valid_positions)
                
        except Exception as e:
            logger.exception("Error updating metrics")
            # Ensure metrics maintain valid values even on error
            self.evaluation_metrics.update({
                'returns': [],
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'win_rate': 0.0
            })
            
    @type_check
    def get_metrics(self) -> Dict[str, Union[float, List[float], int]]:
        """
        Get current evaluation metrics.
        
        Returns:
            Dictionary containing evaluation metrics with specific types:
            - 'returns': List[float]
            - 'sharpe_ratio': float
            - 'max_drawdown': float
            - 'total_trades': int
            - 'win_rate': float
        """
        return self.evaluation_metrics.copy()
        
    @type_check
    def save(self, path: str) -> None:
        """
        Save the model to the specified path.
        
        Args:
            path: Path where the model should be saved (must be a valid file path)
            
        Raises:
            ValueError: If path is empty or invalid
            TypeError: If path is not a string
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        if not path.strip():
            raise ValueError("path cannot be empty")
        self.model.save(path)
        
    @type_check
    def load(self, path: str) -> None:
        """
        Load the model from the specified path.
        
        Args:
            path: Path from where the model should be loaded (must exist)
            
        Raises:
            ValueError: If path is empty or file doesn't exist
            TypeError: If path is not a string
        """
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        if not path.strip():
            raise ValueError("path cannot be empty")
        self.model = PPO.load(path)
