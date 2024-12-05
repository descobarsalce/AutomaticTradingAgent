import os
import numpy as np
from typing import Optional, Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from callbacks import ProgressBarCallback
import gymnasium as gym

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
        Initialize the trading agent with basic configuration.
        
        Args:
            env: Gymnasium environment
            ppo_params: Optional PPO algorithm parameters
            policy_type: Type of policy network to use
            policy_kwargs: Optional policy network parameters
            tensorboard_log: Directory for tensorboard logs
            seed: Random seed for reproducibility
        """
        # Create tensorboard log directory if it doesn't exist
        os.makedirs(tensorboard_log, exist_ok=True)
        
        # Validate environment
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("Environment must have continuous action space (Box)")
        
        # Default PPO parameters
        default_params = {
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 1,
            'seed': seed
        }
        
        if ppo_params:
            default_params.update(ppo_params)
            
        try:
            # Initialize environment with monitoring
            print("Action space:", env.action_space)
            print("Observation space:", env.observation_space)
            self.env = DummyVecEnv([lambda: Monitor(env)])
            
            # Initialize PPO model with parameters
            print("Initializing PPO model with parameters:", default_params)
            self.model = PPO(
                policy_type,
                self.env,
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
                **default_params
            )
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def train(self, total_timesteps: int, eval_freq: int = 10000):
        """
        Train the agent.
        
        Args:
            total_timesteps: Total number of timesteps to train
            eval_freq: Evaluation frequency in timesteps
        """
        try:
            # Setup callbacks
            eval_callback = EvalCallback(
                self.env,
                best_model_save_path='./best_model/',
                log_path='./eval_logs/',
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )
            
            progress_callback = ProgressBarCallback(total_timesteps)
            
            # Train the model
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

    def save(self, path: str):
        """Save the model"""
        self.model.save(path)

    def load(self, path: str):
        """Load the model"""
        self.model = PPO.load(path)
