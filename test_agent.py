import gymnasium as gym
import numpy as np
from agent import TradingAgent
import pandas as pd
import unittest

# Create a simple custom environment for testing
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
        """
        Reset the environment to initial state.
        
        Args:
            seed (int, optional): Random seed
            options (dict, optional): Additional options for reset
            
        Returns:
            tuple: (observation, info)
        """
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
        """
        Execute one step in the environment.
        
        Args:
            action (numpy.ndarray): Continuous action in range [-1, 1]
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
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

# Create test environment
env = SimpleTradingEnv()

# Test custom PPO parameters
custom_params = {
    'learning_rate': 0.0003,
    'n_steps': 2048,
    'batch_size': 128,
    'n_epochs': 5,
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

class TestTradingAgent(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.env = SimpleTradingEnv()
        self.custom_params = custom_params

    def test_initialization(self):
        """Test agent initialization with valid parameters"""
        agent = TradingAgent(self.env, ppo_params=self.custom_params)
        self.assertIsNotNone(agent.model)
        self.assertEqual(len(agent.portfolio_history), 0)
        self.assertEqual(len(agent.positions_history), 0)

    def test_initialization_invalid_env(self):
        """Test agent initialization with invalid environment"""
        with self.assertRaises(TypeError):
            TradingAgent("not_an_env")

    def test_initialization_invalid_seed(self):
        """Test agent initialization with invalid seed"""
        with self.assertRaises(ValueError):
            TradingAgent(self.env, seed=-1)

    def test_predict_invalid_observation(self):
        """Test predict method with invalid observation"""
        agent = TradingAgent(self.env)
        with self.assertRaises(TypeError):
            agent.predict([1, 2, 3, 4])  # Not a numpy array

    def test_predict_wrong_shape(self):
        """Test predict method with wrong observation shape"""
        agent = TradingAgent(self.env)
        wrong_shape = np.array([1, 2])
        with self.assertRaises(ValueError):
            agent.predict(wrong_shape)

    def test_update_state_validation(self):
        """Test update_state method input validation"""
        agent = TradingAgent(self.env)
        
        # Test invalid portfolio value type
        with self.assertRaises(TypeError):
            agent.update_state("not_a_number", {})
            
        # Test negative portfolio value
        with self.assertRaises(ValueError):
            agent.update_state(-1000, {})
            
        # Test invalid positions type
        with self.assertRaises(TypeError):
            agent.update_state(1000, "not_a_dict")
            
        # Test invalid position values
        with self.assertRaises(TypeError):
            agent.update_state(1000, {'AAPL': 'not_a_number'})

    def test_save_load_validation(self):
        """Test save and load methods input validation"""
        agent = TradingAgent(self.env)
        
        # Test invalid path type
        with self.assertRaises(TypeError):
            agent.save(123)
            
        # Test empty path
        with self.assertRaises(ValueError):
            agent.save("")
            
        # Test invalid path type for load
        with self.assertRaises(TypeError):
            agent.load(123)
            
        # Test empty path for load
        with self.assertRaises(ValueError):
            agent.load("")

    def test_valid_workflow(self):
        """Test a valid workflow with proper inputs"""
        agent = TradingAgent(self.env)
        
        # Test valid predict
        obs = np.zeros(4, dtype=np.float32)
        action = agent.predict(obs)
        self.assertIsInstance(action, np.ndarray)
        
        # Test valid update_state
        agent.update_state(10000.0, {'AAPL': 0.6, 'GOOGL': 0.4})
        self.assertEqual(len(agent.portfolio_history), 1)
        self.assertEqual(len(agent.positions_history), 1)
        
        # Test metrics
        metrics = agent.get_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn('returns', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)

if __name__ == '__main__':
    unittest.main()
