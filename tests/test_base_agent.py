import numpy as np
import unittest
import pandas as pd
from core.base_agent import BaseAgent
from environment.simple_trading_env import SimpleTradingEnv

class TestBaseAgent(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.test_data = pd.DataFrame({
            'Open': [100] * 100,
            'High': [110] * 100,
            'Low': [90] * 100,
            'Close': [105] * 100,
            'Volume': [1000] * 100
        })
        self.env = SimpleTradingEnv(
            data=self.test_data,
            initial_balance=100000
        )
        self.custom_params = {
            'learning_rate': 3e-4,
            'n_steps': 1024,
            'batch_size': 128,
            'gamma': 0.98
        }

    def test_base_initialization(self):
        """Test base agent initialization"""
        agent = BaseAgent(self.env, ppo_params=self.custom_params)
        self.assertIsNotNone(agent.model)
        self.assertIsNotNone(agent.env)

    def test_initialization_validation(self):
        """Test initialization input validation"""
        # Test invalid environment
        with self.assertRaises(TypeError):
            BaseAgent("not_an_env")
        
        # Test invalid seed
        with self.assertRaises(ValueError):
            BaseAgent(self.env, seed=-1)
        
        # Test invalid PPO parameters
        with self.assertRaises(TypeError):
            BaseAgent(self.env, ppo_params="not_a_dict")

    def test_base_predict_validation(self):
        """Test predict method input validation"""
        agent = BaseAgent(self.env)
        
        # Test invalid observation type
        with self.assertRaises(TypeError):
            agent.predict([1, 2, 3, 4])  # Not a numpy array
        
        # Test wrong shape
        wrong_shape = np.array([1, 2])
        with self.assertRaises(ValueError):
            agent.predict(wrong_shape)

    def test_state_management(self):
        """Test basic state management functionality"""
        agent = BaseAgent(self.env)
        
        # Test invalid portfolio value type
        with self.assertRaises(TypeError):
            agent.update_state("not_a_number", {})
        
        # Test negative portfolio value
        with self.assertRaises(ValueError):
            agent.update_state(-1000, {})
        
        # Test invalid positions type
        with self.assertRaises(TypeError):
            agent.update_state(1000, "not_a_dict")

    def test_model_persistence(self):
        """Test model save/load functionality"""
        agent = BaseAgent(self.env)
        
        # Test invalid path type
        with self.assertRaises(TypeError):
            agent.save(123)
        
        # Test empty path
        with self.assertRaises(ValueError):
            agent.save("")
        
        # Test invalid path for load
        with self.assertRaises(TypeError):
            agent.load(123)

    def test_basic_workflow(self):
        """Test basic agent workflow"""
        agent = BaseAgent(self.env)
        
        # Test observation processing
        obs = np.zeros(7, dtype=np.float32)
        action = agent.predict(obs)
        self.assertIsInstance(action, np.ndarray)
        
        # Test basic state update
        agent.update_state(10000.0, {'TEST': 1.0})
        self.assertEqual(len(agent.portfolio_history), 1)
        
        # Test metrics interface
        metrics = agent.get_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertTrue(all(key in metrics for key in ['returns', 'sharpe_ratio', 'max_drawdown']))

if __name__ == '__main__':
    unittest.main()
