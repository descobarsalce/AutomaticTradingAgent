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
            initial_balance=100000,
            use_position_profit=True,
            use_holding_bonus=True,
            use_trading_penalty=True
        )

    def test_reward_shaping(self):
        """Test that reward shaping encourages holding profitable positions"""
        # Reset environment
        obs, _ = self.env.reset()

        # Buy action (action=1)
        _, reward_buy, _, _, info = self.env.step(1)

        # Hold position for several steps in upward trend (action=0)
        rewards_hold = []
        for _ in range(5):
            _, reward, _, _, _ = self.env.step(0)
            rewards_hold.append(reward)

        # Verify holding bonus increases over time
        self.assertTrue(all(rewards_hold[i] <= rewards_hold[i+1] 
                          for i in range(len(rewards_hold)-1)),
                      "Rewards should increase when holding profitable position")

        # Test selling penalty (action=2)
        _, reward_trade, _, _, _ = self.env.step(2)
        self.assertLess(reward_trade, rewards_hold[-1],
                       "Trading should incur a penalty compared to holding")

    def test_reward_components(self):
        """Test individual reward components"""
        obs, _ = self.env.reset()

        # Test base reward calculation
        _, reward_small_buy, _, _, _ = self.env.step(1)  # Buy
        obs, _ = self.env.reset()
        rewards = []
        for _ in range(5):  # Test buy in upward trend
            _, reward, _, _, _ = self.env.step(1)
            rewards.append(reward)
        self.assertTrue(any(r > 0 for r in rewards),
                       "Rewards should be positive in upward trend")

    def test_discrete_actions(self):
        """Test discrete action space functionality"""
        obs, _ = self.env.reset()

        # Test invalid action
        with self.assertRaises(ValueError):
            self.env.step(3)  # Invalid action (only 0,1,2 are valid)

        # Test hold action
        initial_shares = self.env.shares_held
        initial_balance = self.env.balance
        _, _, _, _, info = self.env.step(0)
        self.assertEqual(self.env.shares_held, initial_shares)
        self.assertEqual(self.env.balance, initial_balance)

        # Test buy action
        _, _, _, _, info = self.env.step(1)
        self.assertGreater(self.env.shares_held, 0)
        self.assertLess(self.env.balance, initial_balance)

        # Test sell action
        shares_before_sell = self.env.shares_held
        _, _, _, _, info = self.env.step(2)
        self.assertLess(self.env.shares_held, shares_before_sell)

if __name__ == '__main__':
    unittest.main()