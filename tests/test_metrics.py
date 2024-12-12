import unittest
import numpy as np
from core.metrics import (
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_information_ratio,
    calculate_maximum_drawdown
)

class TestMetricsCalculator(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.portfolio_history = [1000, 1100, 1050, 1200, 1150]
        self.returns = np.array([0.1, -0.045, 0.143, -0.042])

    def test_calculate_returns(self):
        """Test returns calculation"""
        returns = calculate_returns(self.portfolio_history)
        self.assertIsInstance(returns, np.ndarray)
        self.assertTrue(len(returns) > 0)
        
    def test_calculate_returns_insufficient_data(self):
        """Test returns calculation with insufficient data"""
        returns = calculate_returns([1000])
        self.assertEqual(len(returns), 0)

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        # Generate more data points for meaningful Sharpe ratio
        returns = np.random.normal(0.001, 0.02, 253)  # One year of daily returns
        sharpe = calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe, float)
        
    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation"""
        # Generate more data points for meaningful Sortino ratio
        returns = np.random.normal(0.001, 0.02, 253)  # One year of daily returns
        sortino = calculate_sortino_ratio(returns)
        self.assertIsInstance(sortino, float)
        
    def test_calculate_information_ratio(self):
        """Test Information ratio calculation"""
        # Generate more data points for meaningful Information ratio
        returns = np.random.normal(0.001, 0.02, 253)  # One year of daily returns
        benchmark_returns = np.random.normal(0.0005, 0.015, 253)
        info_ratio = calculate_information_ratio(returns, benchmark_returns)
        self.assertIsInstance(info_ratio, float)
        
    def test_calculate_maximum_drawdown(self):
        """Test maximum drawdown calculation"""
        max_dd = calculate_maximum_drawdown(self.portfolio_history)
        self.assertIsInstance(max_dd, float)
        self.assertTrue(0 <= max_dd <= 1)

    def test_invalid_inputs(self):
        """Test metrics calculations with invalid inputs"""
        # Test with empty list
        self.assertEqual(len(calculate_returns([])), 0)
        self.assertEqual(calculate_sharpe_ratio(np.array([])), 0.0)
        self.assertEqual(calculate_sortino_ratio(np.array([])), 0.0)
        self.assertEqual(calculate_information_ratio(np.array([])), 0.0)
        self.assertEqual(calculate_maximum_drawdown([]), 0.0)

if __name__ == '__main__':
    unittest.main()
