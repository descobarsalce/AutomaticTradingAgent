import unittest
import numpy as np
import pandas as pd
from datetime import datetime
from utils.common import (
    validate_numeric,
    validate_dataframe,
    calculate_returns,
    format_timestamp,
    validate_trading_params,
    calculate_beta,
    calculate_volatility,
    validate_portfolio_weights,
    format_money
)

class TestUtils(unittest.TestCase):
    def test_validate_numeric(self):
        """Test numeric validation function"""
        self.assertTrue(validate_numeric(100))
        self.assertTrue(validate_numeric(0, allow_zero=True))
        self.assertFalse(validate_numeric(0, allow_zero=False))
        self.assertTrue(validate_numeric(5, min_value=0, max_value=10))
        self.assertFalse(validate_numeric(-1, min_value=0))
        self.assertFalse(validate_numeric(11, max_value=10))
        self.assertFalse(validate_numeric("not a number"))
        
    def test_calculate_volatility(self):
        """Test volatility calculation"""
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        vol = calculate_volatility(returns, annualize=False)
        self.assertIsInstance(vol, float)
        self.assertTrue(vol > 0)
        
        # Test empty array
        self.assertEqual(calculate_volatility(np.array([])), 0.0)
        
        # Test annualization
        annual_vol = calculate_volatility(returns, annualize=True)
        self.assertTrue(annual_vol > vol)  # Annualized should be larger
        
    def test_calculate_beta(self):
        """Test beta calculation"""
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        market_returns = np.array([0.005, -0.01, 0.015, -0.005, 0.01])
        beta = calculate_beta(returns, market_returns)
        self.assertIsInstance(beta, float)
        
        # Test mismatched arrays
        self.assertEqual(calculate_beta(returns, market_returns[:-1]), 0.0)
        
        # Test empty arrays
        self.assertEqual(calculate_beta(np.array([]), np.array([])), 0.0)
        
    def test_validate_portfolio_weights(self):
        """Test portfolio weights validation"""
        valid_weights = {'AAPL': 0.4, 'GOOGL': 0.6}
        self.assertTrue(validate_portfolio_weights(valid_weights))
        
        # Test invalid sum
        invalid_weights = {'AAPL': 0.5, 'GOOGL': 0.6}
        self.assertFalse(validate_portfolio_weights(invalid_weights))
        
        # Test negative weights
        negative_weights = {'AAPL': -0.1, 'GOOGL': 1.1}
        self.assertFalse(validate_portfolio_weights(negative_weights))
        
        # Test empty dict
        self.assertFalse(validate_portfolio_weights({}))
        
    def test_format_money(self):
        """Test money formatting"""
        self.assertEqual(format_money(1234.5678), '$1,234.57')
        self.assertEqual(format_money(0), '$0.00')
        self.assertEqual(format_money(-1234.56), '-$1,234.56')
        self.assertEqual(format_money(1234.5678, '€'), '€1,234.57')

    def test_validate_dataframe(self):
        """Test DataFrame validation function"""
        # Valid DataFrame
        df = pd.DataFrame({
            'Open': [100, 101],
            'Close': [101, 102]
        })
        self.assertTrue(validate_dataframe(df, ['Open', 'Close']))
        
        # Missing columns
        self.assertFalse(validate_dataframe(df, ['Open', 'Close', 'Volume']))
        
        # Empty DataFrame
        empty_df = pd.DataFrame()
        self.assertFalse(validate_dataframe(empty_df, ['Open']))
        
        # Not a DataFrame
        self.assertFalse(validate_dataframe([1, 2, 3], ['Open']))

    def test_calculate_returns(self):
        """Test returns calculation function"""
        values = np.array([100, 110, 105, 115])
        expected_returns = np.array([0.1, -0.0454545, 0.0952381])
        np.testing.assert_almost_equal(
            calculate_returns(values),
            expected_returns,
            decimal=6
        )
        
        # Empty array
        self.assertEqual(len(calculate_returns(np.array([]))), 0)
        
        # Single value
        self.assertEqual(len(calculate_returns(np.array([100]))), 0)

    def test_format_timestamp(self):
        """Test timestamp formatting function"""
        # Test datetime object
        dt = datetime(2024, 1, 1, 12, 30, 45)
        self.assertEqual(
            format_timestamp(dt),
            "2024-01-01 12:30:45"
        )
        
        # Test string timestamp
        self.assertEqual(
            format_timestamp("2024-01-01 12:30:45"),
            "2024-01-01 12:30:45"
        )
        
        # Test invalid timestamp
        self.assertEqual(format_timestamp("invalid"), "")

    def test_validate_trading_params(self):
        """Test trading parameters validation function"""
        # Valid parameters
        valid_params = {
            'learning_rate': 0.001,
            'n_steps': 1024,
            'batch_size': 64
        }
        self.assertTrue(validate_trading_params(valid_params))
        
        # Missing required parameter
        invalid_params = {
            'learning_rate': 0.001,
            'n_steps': 1024
        }
        self.assertFalse(validate_trading_params(invalid_params))
        
        # Invalid learning rate
        invalid_params = {
            'learning_rate': 2.0,  # Should be < 1
            'n_steps': 1024,
            'batch_size': 64
        }
        self.assertFalse(validate_trading_params(invalid_params))
        
        # Invalid steps (negative)
        invalid_params = {
            'learning_rate': 0.001,
            'n_steps': -1024,
            'batch_size': 64
        }
        self.assertFalse(validate_trading_params(invalid_params))
        
        # Invalid batch size (zero)
        invalid_params = {
            'learning_rate': 0.001,
            'n_steps': 1024,
            'batch_size': 0
        }
        self.assertFalse(validate_trading_params(invalid_params))

if __name__ == '__main__':
    unittest.main()
