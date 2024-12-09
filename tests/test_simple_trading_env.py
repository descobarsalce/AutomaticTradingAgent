import unittest
import numpy as np
import pandas as pd
from environment import SimpleTradingEnv

class TestSimpleTradingEnv(unittest.TestCase):
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
            initial_balance=100000
        )

    def test_reward_shaping(self):
        """Test that reward shaping encourages holding profitable positions"""
        # Reset environment
        obs, _ = self.env.reset()
        
        # Buy action (0.5 = 50% of balance)
        _, reward_buy, _, _, info = self.env.step(np.array([0.5]))
        
        # Hold position for several steps in upward trend
        rewards_hold = []
        for _ in range(5):
            _, reward, _, _, _ = self.env.step(np.array([0]))
            rewards_hold.append(reward)
        
        # Verify holding bonus increases over time
        self.assertTrue(all(rewards_hold[i] <= rewards_hold[i+1] 
                          for i in range(len(rewards_hold)-1)),
                      "Rewards should increase when holding profitable position")
        
        # Test trading penalty
        _, reward_trade, _, _, _ = self.env.step(np.array([-0.5]))
        self.assertLess(reward_trade, rewards_hold[-1],
                       "Trading should incur a penalty compared to holding")

    def test_reward_components(self):
        """Test individual reward components"""
        obs, _ = self.env.reset()
        
        # Test base reward calculation
        _, reward_small_buy, _, _, _ = self.env.step(np.array([0.1]))
        _, reward_large_buy, _, _, _ = self.env.step(np.array([0.5]))
        self.assertGreater(abs(reward_large_buy), abs(reward_small_buy),
                          "Larger position changes should have bigger impact")
        
        # Test market trend component
        obs, _ = self.env.reset()
        rewards = []
        for _ in range(5):  # Collect rewards in upward trend
            _, reward, _, _, _ = self.env.step(np.array([0.1]))
            rewards.append(reward)
        self.assertTrue(any(r > 0 for r in rewards),
                       "Rewards should be positive in upward trend")

if __name__ == '__main__':
    unittest.main()
