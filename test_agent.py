import numpy as np
import unittest
from core import TradingAgent
from environment import SimpleTradingEnv

# Create test environment
env = SimpleTradingEnv()

# Test custom PPO parameters
custom_params = {
    'learning_rate': 3e-4,  # Within range (5e-5, 5e-4)
    'n_steps': 1024,        # Within range (512, 2048)
    'batch_size': 128,      # Within range (64, 256)
    'n_epochs': 5,          # Within range (3, 10)
    'gamma': 0.98,          # Within range (0.95, 0.999)
    'gae_lambda': 0.95,     # Within range (0.9, 0.99)
    'clip_range': 0.2,      # Within range (0.1, 0.3)
    'ent_coef': 0.01,       # Within range (0.001, 0.02)
    'vf_coef': 0.6,         # Within range (0.4, 0.9)
    'max_grad_norm': 0.5,   # Within range (0.3, 0.8)
    'use_sde': True,        # Boolean parameter
    'sde_sample_freq': 4,   # Integer parameter
    'target_kl': 0.02       # Within range (0.01, 0.03)
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