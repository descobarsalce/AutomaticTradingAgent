import gymnasium as gym
import numpy as np

class SimpleTradingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Define action space as continuous values between -1 and 1
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        # Define observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )
        self.current_step = 0
        self.max_steps = 100
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        try:
            super().reset(seed=seed)
            self.current_step = 0
            
            # Generate initial observation
            observation = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            
            info = {
                'initial_state': True,
                'step': self.current_step
            }
            
            return observation, info
            
        except Exception as e:
            print(f"Error in reset method: {str(e)}")
            raise
        
    def step(self, action):
        """Execute one step in the environment."""
        try:
            self.current_step += 1
            terminated = self.current_step >= self.max_steps
            truncated = False
            
            # Validate action
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            if not np.all(np.logical_and(action >= -1, action <= 1)):
                raise ValueError("Action values must be in range [-1, 1]")
            
            # Use continuous action directly
            action_value = float(action.flatten()[0])  # Extract single action value
            reward = action_value  # Simple reward based on action
            
            # Generate next state
            next_state = np.array(np.random.randn(4), dtype=np.float32)
            
            info = {
                'step': self.current_step,
                'action_value': action_value
            }
            
            return next_state, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"Error in step method: {str(e)}")
            raise
