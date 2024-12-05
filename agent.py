import numpy as np
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm
from stable_baselines3.common.monitor import Monitor
import os
import torch

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
        # Create tensorboard log directory if it doesn't exist
        os.makedirs(tensorboard_log, exist_ok=True)
        
        # Initialize enhanced state tracking
        self.portfolio_value = 1.0
        self.positions = {}
        self.trades_history = []
        self.action_history = []
        self.reward_history = []
        self.state_history = []
        self.evaluation_metrics = {
            'returns': [],
            'sharpe_ratio': None,
            'max_drawdown': None,
            'sortino_ratio': None,
            'win_rate': None,
            'profit_factor': None,
            'volatility': None,
            'beta': None,
            'alpha': None
        }
        
        # Track training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': []
        }
        
        # Default PPO parameters with advanced configuration
        default_params = {
            'learning_rate': 0.00025,
            'n_steps': 1024,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'clip_range_vf': None,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'use_sde': False,
            'sde_sample_freq': -1,
            'target_kl': 0.015,
            'verbose': 1,
            'seed': seed
        }
        
        # Default policy network configuration
        default_policy_kwargs = {
            'net_arch': dict(pi=[64, 64], vf=[64, 64]),
            'activation_fn': torch.nn.Tanh,
            'ortho_init': True
        }
        
        # Update with custom parameters if provided
        if ppo_params:
            default_params.update(ppo_params)
        if policy_kwargs:
            default_policy_kwargs.update(policy_kwargs)
        
        try:
            # Initialize environment with monitoring
            print("Initializing environment with monitoring...")
            self.env = DummyVecEnv([lambda: Monitor(env)])
            
            # Validate action and observation spaces
            print(f"Action space: {self.env.action_space}")
            print(f"Observation space: {self.env.observation_space}")
            
            # Initialize PPO model with enhanced parameters
            print("Initializing PPO model with parameters:", default_params)
            self.model = PPO(
                policy_type,
                self.env,
                tensorboard_log=tensorboard_log,
                policy_kwargs=default_policy_kwargs,
                **default_params
            )
            print("PPO model initialized successfully")
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise
    
    def train(
        self,
        total_timesteps: int,
        eval_freq: int = 10000,
        early_stopping_patience: Optional[int] = None,
        target_reward: Optional[float] = None
    ):
        """
        Train the agent with enhanced evaluation and early stopping
        
        Args:
            total_timesteps: Total number of timesteps to train
            eval_freq: Evaluation frequency in timesteps
            early_stopping_patience: Number of evaluations without improvement before stopping
            target_reward: Target mean reward to stop training
        """
        callbacks = []
        
        # Setup evaluation callback
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path='./best_model/',
            log_path='./eval_logs/',
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # Add reward threshold callback if target specified
        if target_reward is not None:
            stop_on_reward = StopTrainingOnRewardThreshold(
                reward_threshold=target_reward,
                verbose=1
            )
            eval_callback = EvalCallback(
                self.env,
                callback_on_new_best=stop_on_reward,
                eval_freq=eval_freq,
                best_model_save_path='./best_model/',
                verbose=1
            )
            callbacks.append(eval_callback)
        
        # Add progress bar callback
        from callbacks import ProgressBarCallback
        progress_callback = ProgressBarCallback(total_timesteps)
        callbacks.append(progress_callback)
        
        # Train the model with enhanced callbacks
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks
        )
        
        # Update training metrics
        rollout_buffer = self.model.rollout_buffer
        if rollout_buffer is not None:
            self.training_metrics['episode_rewards'].extend(
                self.model.ep_info_buffer
            )
            self.training_metrics['policy_loss'].append(
                float(self.model.logger.name_to_value['train/policy_gradient_loss'])
            )
            self.training_metrics['value_loss'].append(
                float(self.model.logger.name_to_value['train/value_loss'])
            )
            self.training_metrics['entropy_loss'].append(
                float(self.model.logger.name_to_value['train/entropy_loss'])
            )
    
    def predict(self, observation, deterministic: bool = True):
        """
        Make a prediction and update state history
        """
        action, _states = self.model.predict(
            observation,
            deterministic=deterministic
        )
        # Track state and action history
        self.state_history.append(observation)
        self.action_history.append(action)
        return action
    
    def update_state(self, portfolio_value: float, positions: Dict[str, float]):
        """
        Update the agent's state tracking
        """
        self.portfolio_value = portfolio_value
        self.positions = positions
        self.trades_history.append({
            'portfolio_value': portfolio_value,
            'positions': positions.copy()
        })
    
    def calculate_metrics(self, risk_free_rate: float = 0.02):
        """
        Calculate and update comprehensive evaluation metrics
        
        Args:
            risk_free_rate: Annual risk-free rate for ratio calculations
        """
        if len(self.trades_history) < 2:
            return
        
        # Calculate returns
        portfolio_values = np.array([t['portfolio_value'] for t in self.trades_history])
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
        
        if len(returns) > 1:
            # Basic return metrics
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Sharpe ratio
            excess_returns = returns - daily_rf_rate
            sharpe_ratio = np.mean(excess_returns) / std_return * np.sqrt(252)
            self.evaluation_metrics['sharpe_ratio'] = sharpe_ratio
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                sortino_ratio = (mean_return - daily_rf_rate) / downside_std * np.sqrt(252)
                self.evaluation_metrics['sortino_ratio'] = sortino_ratio
            
            # Win rate
            winning_trades = np.sum(returns > 0)
            self.evaluation_metrics['win_rate'] = winning_trades / len(returns)
            
            # Profit factor
            gross_profits = np.sum(returns[returns > 0])
            gross_losses = abs(np.sum(returns[returns < 0]))
            self.evaluation_metrics['profit_factor'] = gross_profits / gross_losses if gross_losses != 0 else float('inf')
            
            # Volatility (annualized)
            self.evaluation_metrics['volatility'] = std_return * np.sqrt(252)
        
        # Calculate maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdowns = (peak - portfolio_values) / peak
        self.evaluation_metrics['max_drawdown'] = np.max(drawdowns)
        
        # Store processed returns
        self.evaluation_metrics['returns'] = returns.tolist()
    
    def save(self, path: str):
        """
        Save the model and state
        """
        self.model.save(path)
        # Save additional state information if needed
        state_path = f"{path}_state"
        np.save(state_path, {
            'portfolio_value': self.portfolio_value,
            'positions': self.positions,
            'evaluation_metrics': self.evaluation_metrics
        })
    
    def load(self, path: str):
        """
        Load the model and state
        """
        self.model = PPO.load(path)
        # Load additional state information if available
        state_path = f"{path}_state.npy"
        if os.path.exists(state_path):
            state_dict = np.load(state_path, allow_pickle=True).item()
            self.portfolio_value = state_dict['portfolio_value']
            self.positions = state_dict['positions']
            self.evaluation_metrics = state_dict['evaluation_metrics']
