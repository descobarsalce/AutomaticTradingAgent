"""
Trading platform formatting utility functions.
Provides string formatting and display functions.
"""

from datetime import datetime
import pandas as pd
from typing import Union
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def format_timestamp(timestamp: Union[str, datetime]) -> str:
    """Format timestamp to standard string format."""
    try:
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        logger.error(f"Error formatting timestamp: {str(e)}")
        return ""

def format_money(value: float, currency: str = '$') -> str:
    """Format monetary values with proper separators and currency symbol."""
    try:
        if value < 0:
            return f"-{currency}{abs(value):,.2f}"
        return f"{currency}{value:,.2f}"
    except Exception as e:
        logger.error(f"Error formatting monetary value: {str(e)}")
        return f"{currency}0.00"

def format_date(date: Union[str, datetime], format_str: str = "%Y-%m-%d") -> str:
    """Format date string or datetime object to specified format."""
    try:
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return date.strftime(format_str)
    except Exception as e:
        logger.error(f"Error formatting date: {str(e)}")
        return ""

def round_price(price: float, precision: int = 2) -> float:
    """Round price to specified precision."""
    try:
        return round(float(price), precision)
    except Exception as e:
        logger.error(f"Error rounding price: {str(e)}")
        return 0.0
