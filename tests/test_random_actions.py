import unittest
import numpy as np
import pandas as pd
from environment import SimpleTradingEnv

class TestRandomActions(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        # Create test data with an upward trend
        self.test_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104] * 20,
            'High': [105, 106, 107, 108, 109] * 20,
            'Low': [98, 99, 100, 101, 102] * 20,
            'Close': [102, 103, 104, 105, 106] * 20,
            'Volume': [1000] * 100
        })
        self.env = SimpleTradingEnv(
            data=self.test_data,
            initial_balance=100000,
            training_mode=True  # Enable training mode for exploration
        )

    def test_random_action_execution(self):
        """Test that random actions produce reasonable behavior"""
        n_steps = 100
        stats = self.env.test_random_actions(n_steps)
        
        # Verify all action types are used
        self.assertTrue(all(count > 0 for count in stats['actions_taken'].values()),
                       "All action types should be used in random testing")
        
        # Verify some trades are executed
        self.assertGreater(stats['trades_executed'], 0,
                          "Some trades should be executed during random testing")
        
        # Verify trade success rate is reasonable (above 0.5 with reduced transaction costs)
        self.assertGreater(stats['trade_success_rate'], 0.5,
                          "Trade success rate should be reasonable in training mode")

    def test_training_mode_benefits(self):
        """Test that training mode provides benefits for exploration"""
        # Create environments with and without training mode
        env_training = SimpleTradingEnv(
            data=self.test_data,
            initial_balance=100000,
            training_mode=True
        )
        env_normal = SimpleTradingEnv(
            data=self.test_data,
            initial_balance=100000,
            training_mode=False
        )
        
        # Test both environments
        stats_training = env_training.test_random_actions(100)
        stats_normal = env_normal.test_random_actions(100)
        
        # Training mode should have higher success rate due to reduced costs
        self.assertGreater(stats_training['trade_success_rate'],
                          stats_normal['trade_success_rate'],
                          "Training mode should have higher trade success rate")
        
        # Training mode should have higher average reward due to exploration bonus
        self.assertGreater(stats_training['avg_reward'],
                          stats_normal['avg_reward'],
                          "Training mode should have higher average reward")

if __name__ == '__main__':
    unittest.main()
