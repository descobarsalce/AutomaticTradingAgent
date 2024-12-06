import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from callbacks import ProgressBarCallback
from numpy.typing import NDArray

class TradingAgent:
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
        """
        Initialize the trading agent with advanced configuration and state tracking.
        
        Args:
            env: Gymnasium environment
            ppo_params: Optional PPO algorithm parameters
            policy_type: Type of policy network to use
            policy_kwargs: Optional policy network parameters
            tensorboard_log: Directory for tensorboard logs
            seed: Random seed for reproducibility
        """
        # Initialize state tracking
        self.portfolio_history = []
        self.positions_history = []
        self.evaluation_metrics = {
            'returns': [],
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'win_rate': 0.0
        }
        
        # Set up default PPO parameters if none provided
        if ppo_params is None:
            ppo_params = {
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
            
        # Set up default policy network parameters if none provided
        if policy_kwargs is None:
            policy_kwargs = {
                'net_arch': [dict(pi=[64, 64], vf=[64, 64])]
            }
            
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
            print(f"Error initializing PPO model: {str(e)}")
            raise
            
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
            print(f"Error during training: {str(e)}")
            raise
            
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
            print(f"Error in predict method: {str(e)}")
            raise
            
    def update_state(self, portfolio_value: float, positions: Dict[str, float]) -> None:
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
        
    def _update_metrics(self):
        """Update evaluation metrics based on current state"""
        if len(self.portfolio_history) > 1:
            # Calculate returns
            returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
            self.evaluation_metrics['returns'] = returns.tolist()
            
            # Calculate Sharpe ratio (assuming daily returns)
            if len(returns) > 1:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                self.evaluation_metrics['sharpe_ratio'] = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            
            # Calculate maximum drawdown
            peak = self.portfolio_history[0]
            max_dd = 0
            for value in self.portfolio_history[1:]:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
            self.evaluation_metrics['max_drawdown'] = max_dd
            
            # Update trade statistics
            self.evaluation_metrics['total_trades'] = len(self.positions_history)
            
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
