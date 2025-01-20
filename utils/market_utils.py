"""
Trading platform market utility functions.
Provides market analysis, calculations, and trading-related functions.
"""

from datetime import datetime
import logging
from typing import Union
import pandas as pd
from metrics.metrics_calculator import MetricsCalculator

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def is_market_hours(timestamp: Union[str, datetime], market_open: str = "09:30", market_close: str = "16:00") -> bool:
    """Check if given timestamp is during market hours."""
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
    """Calculate number of trading days between two dates."""
    try:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        date_range = pd.date_range(start_date, end_date)
        trading_days = len(date_range[date_range.dayofweek < 5])

        return trading_days
    except Exception as e:
        logger.error(f"Error calculating trading days: {str(e)}")
        return 0