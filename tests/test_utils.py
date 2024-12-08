import unittest
import numpy as np
import pandas as pd
from datetime import datetime
from utils.data_utils import (
    validate_numeric,
    validate_dataframe,
    validate_portfolio_weights,
    normalize_data
)
from utils.market_utils import (
    calculate_returns,
    calculate_beta,
    calculate_volatility,
    calculate_moving_average,
    calculate_ema,
    calculate_correlation,
    calculate_bollinger_bands,
    calculate_rsi,
    calculate_macd
)
from utils.formatting_utils import (
    format_timestamp,
    format_money,
    format_date,
    round_price
)
from utils.common import (
    validate_trading_params,
    MAX_POSITION_SIZE,
    MIN_POSITION_SIZE
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


    def test_calculate_moving_average(self):
        """Test moving average calculation"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Test with window size 3
        ma = calculate_moving_average(data, window=3)
        self.assertEqual(len(ma), len(data) - 2)  # Valid convolution length
        self.assertAlmostEqual(ma[0], 2.0)  # (1 + 2 + 3) / 3
        
        # Test empty data
        self.assertEqual(len(calculate_moving_average(np.array([]))), 0)
        
        # Test data smaller than window
        self.assertEqual(len(calculate_moving_average(np.array([1, 2]), window=3)), 0)
        
    def test_calculate_ema(self):
        """Test exponential moving average calculation"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Test EMA calculation
        ema = calculate_ema(data, span=3)
        self.assertEqual(len(ema), len(data))
        self.assertTrue(np.all(ema[1:] > ema[:-1]))  # EMA should be monotonically increasing
        
        # Test empty data
        self.assertEqual(len(calculate_ema(np.array([]))), 0)
        
        # Test data smaller than span
        self.assertEqual(len(calculate_ema(np.array([1, 2]), span=3)), 0)
        
    def test_calculate_correlation(self):
        """Test correlation calculation"""
        series1 = np.array([1, 2, 3, 4, 5])
        series2 = np.array([2, 4, 6, 8, 10])
        
        # Perfect positive correlation
        corr = calculate_correlation(series1, series2)
        self.assertAlmostEqual(corr, 1.0)
        
        # Perfect negative correlation
        corr = calculate_correlation(series1, -series2)
        self.assertAlmostEqual(corr, -1.0)
        
        # No correlation
        corr = calculate_correlation(series1, np.zeros_like(series1))
        self.assertEqual(corr, 0.0)
        
        # Different lengths
        self.assertEqual(calculate_correlation(series1, series2[:-1]), 0.0)
        
    def test_normalize_data(self):
        """Test data normalization"""
        data = np.array([1, 2, 3, 4, 5])
        
        # Test min-max normalization
        minmax = normalize_data(data, method='minmax')
        self.assertAlmostEqual(np.min(minmax), 0.0)
        self.assertAlmostEqual(np.max(minmax), 1.0)
        
        # Test z-score normalization
        zscore = normalize_data(data, method='zscore')
        self.assertAlmostEqual(np.mean(zscore), 0.0, places=7)
        self.assertAlmostEqual(np.std(zscore), 1.0, places=7)
        
        # Test invalid method
        self.assertEqual(len(normalize_data(data, method='invalid')), 0)
        
        # Test empty data
        self.assertEqual(len(normalize_data(np.array([]))), 0)
        
        # Test constant data
        const_data = np.array([1, 1, 1, 1])
        self.assertTrue(np.all(normalize_data(const_data) == 0))
if __name__ == '__main__':

    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3)  # Repeated sequence for more data points
        
        # Test normal calculation
        middle, upper, lower = calculate_bollinger_bands(data, window=5, num_std=2.0)
        self.assertTrue(len(middle) > 0)
        self.assertTrue(np.all(upper >= middle))
        self.assertTrue(np.all(middle >= lower))
        
        # Test insufficient data
        middle, upper, lower = calculate_bollinger_bands(np.array([1, 2]), window=5)
        self.assertEqual(len(middle), 0)
        self.assertEqual(len(upper), 0)
        self.assertEqual(len(lower), 0)
        
    def test_calculate_rsi(self):
        """Test RSI calculation"""
        data = np.array([10, 12, 11, 13, 15, 14, 16, 18, 17, 19])
        
        # Test normal calculation
        rsi = calculate_rsi(data, period=3)
        self.assertTrue(len(rsi) > 0)
        self.assertTrue(np.all((rsi >= 0) & (rsi <= 100)))
        
        # Test insufficient data
        self.assertEqual(len(calculate_rsi(np.array([1, 2]), period=3)), 0)
        
    def test_calculate_macd(self):
        """Test MACD calculation"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 4)  # Repeated sequence for more data points
        
        # Test normal calculation
        macd_line, signal_line = calculate_macd(data, fast_period=3, slow_period=6, signal_period=2)
        self.assertTrue(len(macd_line) > 0)
        self.assertTrue(len(signal_line) > 0)
        self.assertEqual(len(macd_line), len(signal_line))
        
        # Test insufficient data
        macd_line, signal_line = calculate_macd(np.array([1, 2, 3]))
        self.assertEqual(len(macd_line), 0)
        self.assertEqual(len(signal_line), 0)
    unittest.main()
