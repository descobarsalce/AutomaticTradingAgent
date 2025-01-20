
"""
Trading platform shared utilities module.
Contains common constants and validation functions used across the platform.
"""

from typing import Dict, Any, Union, Optional, Callable
import logging
import inspect
import numpy as np
from datetime import datetime
import pandas as pd
from gymnasium import Env
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add the handlers to the logger if not already added
if not logger.handlers:
    logger.addHandler(ch)

# Trading Constants
MAX_POSITION_SIZE = 100.0
MIN_POSITION_SIZE = -100.0
MAX_LEVERAGE = 5.0
MIN_TRADE_SIZE = 0.01
DEFAULT_STOP_LOSS = 1
DEFAULT_TAKE_PROFIT = 1
TRADING_DAYS_PER_YEAR = 252
MIN_DATA_POINTS = 252
RISK_FREE_RATE = 0.02
MAX_TRADES_PER_DAY = 10
CORRELATION_THRESHOLD = 0.7
PRICE_PRECISION = 2
POSITION_PRECISION = 4
ANNUALIZATION_FACTOR = 252 ** 0.5

def validate_trading_params(params: Dict[str, Any]) -> bool:
    try:
        required_params = {'learning_rate', 'n_steps', 'batch_size'}
        if not all(param in params for param in required_params):
            return False
        if not (0 < params['learning_rate'] < 1):
            return False
        if not (isinstance(params['n_steps'], int) and params['n_steps'] > 0):
            return False
        if not (isinstance(params['batch_size'], int) and params['batch_size'] > 0):
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating trading parameters: {str(e)}")
        return False

def validate_numeric(value: Union[int, float], min_value: Optional[float] = None, 
                    max_value: Optional[float] = None, allow_zero: bool = True) -> bool:
    try:
        if not isinstance(value, (int, float)):
            return False
        if not allow_zero and value == 0:
            return False
        if min_value is not None and value < min_value:
            return False
        if max_value is not None and value > max_value:
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating numeric value: {str(e)}")
        return False

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    try:
        if not isinstance(df, pd.DataFrame):
            return False
        if df.empty:
            return False
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating DataFrame: {str(e)}")
        return False

def validate_portfolio_weights(weights: Dict[str, float]) -> bool:
    try:
        if not weights:
            return False
        if not all(isinstance(w, (int, float)) for w in weights.values()):
            return False
        if not all(0 <= w <= 1 for w in weights.values()):
            return False
        return abs(sum(weights.values()) - 1.0) < 1e-6
    except Exception as e:
        logger.error(f"Error validating portfolio weights: {str(e)}")
        return False

def format_timestamp(timestamp: Union[str, datetime]) -> str:
    try:
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        logger.error(f"Error formatting timestamp: {str(e)}")
        return ""

def format_money(value: float, currency: str = '$') -> str:
    try:
        if value < 0:
            return f"-{currency}{abs(value):,.2f}"
        return f"{currency}{value:,.2f}"
    except Exception as e:
        logger.error(f"Error formatting monetary value: {str(e)}")
        return f"{currency}0.00"

def format_date(date: Union[str, datetime], format_str: str = "%Y-%m-%d") -> str:
    try:
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return date.strftime(format_str)
    except Exception as e:
        logger.error(f"Error formatting date: {str(e)}")
        return ""

def round_price(price: float, precision: int = 2) -> float:
    try:
        return round(float(price), precision)
    except Exception as e:
        logger.error(f"Error rounding price: {str(e)}")
        return 0.0

def is_market_hours(timestamp: Union[str, datetime], market_open: str = "09:30", market_close: str = "16:00") -> bool:
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

def type_check(func: Callable) -> Callable:
    """Decorator for runtime type checking of function parameters and return value."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        hints = get_type_hints(func)
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        def check_type(value: Any, expected_type: Any) -> bool:
            if value is None:
                if hasattr(expected_type, "__origin__") and expected_type.__origin__ is Union:
                    return type(None) in expected_type.__args__
                return False
            
            if expected_type is np.ndarray:
                return isinstance(value, np.ndarray)
            
            if expected_type is Env:
                return isinstance(value, Env)
            
            if hasattr(expected_type, "__origin__") and expected_type.__origin__ is Union:
                return any(check_type(value, t) for t in expected_type.__args__)
            
            if hasattr(expected_type, "__origin__"):
                return isinstance(value, expected_type.__origin__)
            
            return isinstance(value, expected_type)
            
        for param_name, param_value in bound_args.arguments.items():
            if param_name in hints:
                expected_type = hints[param_name]
                if not check_type(param_value, expected_type):
                    raise TypeError(
                        f"Parameter '{param_name}' must be {expected_type}, "
                        f"got {type(param_value).__name__} instead"
                    )
        
        result = func(*args, **kwargs)
        
        if 'return' in hints and func.__name__ != '__init__':
            return_type = hints['return']
            if return_type is type(None):
                if result is not None:
                    raise TypeError(
                        f"Function should return None, "
                        f"got {type(result).__name__} instead"
                    )
            elif not check_type(result, return_type):
                raise TypeError(
                    f"Function should return {return_type}, "
                    f"got {type(result).__name__} instead"
                )
        
        return result
    
    return wrapper
