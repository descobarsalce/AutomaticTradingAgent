import numpy as np
from typing import Dict, Any, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from callbacks import ProgressBarCallback

class TradingAgent:
    def __init__(
        self,
        env,
        ppo_params: Optional[Dict[str, Any]] = None,
        policy_type: str = "MlpPolicy",
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: str = "./tensorboard_logs/",
        seed: Optional[int] = None
    ):
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
            
    def train(self, total_timesteps: int):
        """
        Train the agent for a specified number of timesteps.
        
        Args:
            total_timesteps: Number of timesteps to train for
        """
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
            
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Make a prediction based on the current observation.
        
        Args:
            observation: Current environment observation
            deterministic: Whether to use deterministic actions
            
        Returns:
            action: Predicted action
        """
        try:
            action, _states = self.model.predict(
                observation,
                deterministic=deterministic
            )
            return action
            
        except Exception as e:
            print(f"Error in predict method: {str(e)}")
            raise
            
    def update_state(self, portfolio_value: float, positions: Dict[str, float]):
        """
        Update agent's state tracking with new portfolio information.
        
        Args:
            portfolio_value: Current portfolio value
            positions: Dictionary of current positions (symbol: weight)
        """
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
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get current evaluation metrics"""
        return self.evaluation_metrics.copy()
        
    def save(self, path: str):
        """Save the model"""
        self.model.save(path)
        
    def load(self, path: str):
        """Load the model"""
        self.model = PPO.load(path)
