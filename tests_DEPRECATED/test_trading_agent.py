import numpy as np
import unittest
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from core import TradingAgent
from environment import SimpleTradingEnv

class TestTradingAgent(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        # Create test data
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

    def test_trading_specific_initialization(self):
        """Test trading-specific initialization"""
        agent = TradingAgent(self.env, ppo_params=self.custom_params)
        self.assertIsNotNone(agent.model)
        # Verify trading-specific attributes
        self.assertEqual(len(agent.portfolio_history), 0)
        self.assertEqual(len(agent.positions_history), 0)
        
    def test_quick_mode_initialization(self):
        """Test quick mode initialization with minimal parameters"""
        agent = TradingAgent(self.env, quick_mode=True)
        self.assertIsNotNone(agent.model)
        self.assertTrue(agent.quick_mode)
        
        # Verify quick mode parameters
        model_params = agent.model.get_parameters()
        learning_rate = float(str(agent.model.learning_rate))  # Convert learning rate to float
        
        self.assertEqual(learning_rate, 3e-4)
        self.assertEqual(agent.model.n_steps, 128)
        self.assertEqual(agent.model.batch_size, 32)

    def test_trading_action_bounds(self):
        """Test that trading actions are properly bounded"""
        agent = TradingAgent(self.env)
        obs = np.zeros(7, dtype=np.float32)  # OHLCV + position + balance
        action = agent.predict(obs)
        self.assertIsInstance(action, np.ndarray)
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))

    def test_portfolio_tracking(self):
        """Test portfolio state tracking functionality"""
        agent = TradingAgent(self.env)
        
        # Test portfolio updates
        test_values = [
            (100000.0, {'AAPL': 0.6}),
            (102000.0, {'AAPL': 0.7}),
            (101000.0, {'AAPL': 0.5})
        ]
        
        for value, positions in test_values:
            agent.update_state(value, positions)
        
        self.assertEqual(len(agent.portfolio_history), 3)
        self.assertEqual(len(agent.positions_history), 3)
        self.assertEqual(agent.portfolio_history[-1], 101000.0)

    def test_trading_metrics(self):
        """Test trading-specific metrics calculation"""
        agent = TradingAgent(self.env)
        
        # Simulate some portfolio updates
        updates = [
            (100000.0, {'AAPL': 0.0}),  # Initial
            (102000.0, {'AAPL': 0.5}),  # Up 2%
            (103000.0, {'AAPL': 0.7}),  # Up 1%
            (101000.0, {'AAPL': 0.3})   # Down 2%
        ]
        
        for value, positions in updates:
            agent.update_state(value, positions)
        
        metrics = agent.get_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn('returns', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        
        # Verify basic metric calculations
        self.assertEqual(metrics['max_drawdown'], 0.02)  # 2% drop from peak
        self.assertTrue(metrics['returns'] > 0)  # Overall positive return

    def test_trading_environment_interaction(self):
        """Test interaction with trading environment"""
        agent = TradingAgent(self.env)
        
        # Test full episode
        obs, info = self.env.reset()
        done = False
        total_steps = 0
        
        while not done and total_steps < 10:  # Limit steps for testing
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_steps += 1
            
            # Verify action and observation shapes
            self.assertEqual(action.shape, (1,))
            self.assertEqual(obs.shape, (7,))
            
            # Verify info dictionary contains required keys
            self.assertIn('net_worth', info)
            self.assertIn('balance', info)
            self.assertIn('shares_held', info)
            
    def test_train_with_callback(self):
        """Test training with callback functionality"""
        agent = TradingAgent(self.env)
        
        # Create a simple test callback
        class TestCallback(BaseCallback):
            def __init__(self):
                super().__init__(verbose=0)
                self.called = False
            
            def _on_step(self):
                self.called = True
                return True
        
        callback = TestCallback()
        
        # Test training with callback
        agent.train(total_timesteps=100, callback=callback)
        self.assertTrue(callback.called, "Callback was not called during training")
        
        # Test training without callback
        agent.train(total_timesteps=100)  # Should not raise any errors
        
        # Test invalid input
        with self.assertRaises(ValueError):
            agent.train(total_timesteps=-1)
            
        with self.assertRaises(ValueError):
            agent.train(total_timesteps=0)