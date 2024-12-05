import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
    CallbackList
)

class TradingAgent:
    def __init__(self, env, model_name="trading_model"):
        self.env = DummyVecEnv([lambda: env])
        self.model_name = model_name
        
        # Create directories for saving models and logs
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            learning_rate=0.00025,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=f"logs/{model_name}"
        )
        
    def setup_callbacks(self, eval_freq=1000, save_freq=1000):
        """Setup training callbacks"""
        # Checkpoint callback - save model periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=f"models/{self.model_name}",
            name_prefix="ppo_trading"
        )
        
        # Evaluation callback
        eval_env = DummyVecEnv([lambda: self.env.envs[0]])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"models/{self.model_name}/best_model",
            log_path=f"logs/{self.model_name}",
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Early stopping callback
        stop_train_callback = StopTrainingOnRewardThreshold(
            reward_threshold=2.0,  # Stop if we reach 200% returns
            verbose=1
        )
        
        return CallbackList([checkpoint_callback, eval_callback, stop_train_callback])
        
    def train(self, total_timesteps):
        """Train the agent with callbacks"""
        callbacks = self.setup_callbacks()
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks
        )
        
    def predict(self, observation):
        action, _states = self.model.predict(observation)
        return action
