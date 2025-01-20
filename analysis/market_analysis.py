"""Market analysis functionality"""
from metrics.metrics_calculator import MetricsCalculator

# Re-export all metrics functions through this module for backward compatibility
calculate_returns = MetricsCalculator.calculate_returns
calculate_volatility = MetricsCalculator.calculate_volatility
calculate_beta = MetricsCalculator.calculate_beta
calculate_bollinger_bands = MetricsCalculator.calculate_bollinger_bands
calculate_rsi = MetricsCalculator.calculate_rsi
calculate_macd = MetricsCalculator.calculate_macd

def is_market_hours(timestamp: Union[str, datetime], market_open: str = "09:30", market_close: str = "16:00") -> bool:
    """Check if within market hours."""
    try:
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        current_time = timestamp.time()
        open_time = datetime.strptime(market_open, "%H:%M").time()
        close_time = datetime.strptime(market_close, "%H:%M").time()
        return open_time <= current_time <= close_time
    except Exception as e:
        logger.error(f"Error checking market hours: {str(e)}")
        return False

def trading_days_between(start_date: Union[str, datetime], end_date: Union[str, datetime]) -> int:
    """Calculate trading days between dates."""
    try:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        bdays = pd.date_range(start=start_date, end=end_date, freq='B')
        return len(bdays)
    except Exception as e:
        logger.error(f"Error calculating trading days: {str(e)}")
        return 0