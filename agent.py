import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class TradingAgent:
    def __init__(self, env):
        self.env = DummyVecEnv([lambda: env])
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
            clip_range=0.2
        )
        
    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)
        
    def predict(self, observation):
        action, _states = self.model.predict(observation)
        return action
