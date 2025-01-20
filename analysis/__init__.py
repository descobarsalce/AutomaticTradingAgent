
from metrics.metrics_calculator import MetricsCalculator
from utils.market_utils import (
    is_market_hours,
    trading_days_between
)

# Re-export metrics functions for backward compatibility
calculate_returns = MetricsCalculator.calculate_returns
calculate_volatility = MetricsCalculator.calculate_volatility
calculate_beta = MetricsCalculator.calculate_beta
calculate_bollinger_bands = MetricsCalculator.calculate_bollinger_bands
calculate_rsi = MetricsCalculator.calculate_rsi
calculate_macd = MetricsCalculator.calculate_macd

__all__ = [
    'calculate_returns',
    'calculate_volatility', 
    'calculate_beta',
    'calculate_bollinger_bands',
    'calculate_rsi',
    'calculate_macd',
    'is_market_hours',
    'trading_days_between'
]
