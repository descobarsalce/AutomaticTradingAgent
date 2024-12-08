"""
Trading platform shared utilities module with common functions and constants.

This module provides:
1. Data validation and formatting functions
2. Technical indicators calculation
3. Data normalization and processing
4. Trading constants and configuration
5. Date/time handling utilities
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Union, Dict, List, Any, Optional
import logging

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

def validate_numeric(value: Union[int, float], min_value: Optional[float] = None, 
                    max_value: Optional[float] = None, allow_zero: bool = True) -> bool:
    """
    Validate numeric values within specified ranges.
    
    Args:
        value: The numeric value to validate
        min_value: Optional minimum value
        max_value: Optional maximum value
        allow_zero: Whether to allow zero values
        
    Returns:
        bool: True if valid, False otherwise
    """
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

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate DataFrame structure and contents.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        bool: True if valid, False otherwise
    """
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

def calculate_returns(values: np.ndarray, round_precision: Optional[int] = None) -> np.ndarray:
    """
    Calculate returns from a series of values.
    
    Args:
        values: Array of values
        round_precision: Optional number of decimal places for rounding
        
    Returns:
        np.ndarray: Calculated returns
    """
    try:
        if len(values) < 2:
            return np.array([])
        returns = np.diff(values) / values[:-1]
        if round_precision is not None:
            returns = np.round(returns, round_precision)
        return returns
    except Exception as e:
        logger.error(f"Error calculating returns: {str(e)}")
        return np.array([])

def format_timestamp(timestamp: Union[str, datetime]) -> str:
    """
    Format timestamp to standard string format.
    
    Args:
        timestamp: Timestamp to format
        
    Returns:
        str: Formatted timestamp string
    """
    try:
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        logger.error(f"Error formatting timestamp: {str(e)}")
        return ""

def validate_trading_params(params: Dict[str, Any]) -> bool:
    """
    Validate trading parameters.
    
    Args:
        params: Dictionary of trading parameters
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        required_params = {'learning_rate', 'n_steps', 'batch_size'}
        if not all(param in params for param in required_params):
            return False
            
        # Validate learning rate
        if not (0 < params['learning_rate'] < 1):
            return False
            
        # Validate steps and batch size
        if not (isinstance(params['n_steps'], int) and params['n_steps'] > 0):
            return False
            
        if not (isinstance(params['batch_size'], int) and params['batch_size'] > 0):
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating trading parameters: {str(e)}")
        return False

def calculate_volatility(returns: np.ndarray, annualize: bool = True) -> float:
    """
    Calculate the volatility of returns.
    
    Args:
        returns: Array of returns
        annualize: Whether to annualize the volatility
        
    Returns:
        float: Calculated volatility
    """
    try:
        if len(returns) < 2:
            return 0.0
        vol = np.std(returns, ddof=1)
        if annualize:
            vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
        return float(vol)
    except Exception as e:
        logger.error(f"Error calculating volatility: {str(e)}")
        return 0.0

def calculate_beta(returns: np.ndarray, market_returns: np.ndarray) -> float:
    """
    Calculate the beta of returns against market returns.
    
    Args:
        returns: Array of asset returns
        market_returns: Array of market returns
        
    Returns:
        float: Calculated beta
    """
    try:
        if len(returns) != len(market_returns) or len(returns) < 2:
            return 0.0
        covariance = np.cov(returns, market_returns)[0][1]
        market_variance = np.var(market_returns, ddof=1)
        return float(covariance / market_variance if market_variance != 0 else 0.0)
    except Exception as e:
        logger.error(f"Error calculating beta: {str(e)}")
        return 0.0

def validate_portfolio_weights(weights: Dict[str, float]) -> bool:
    """
    Validate portfolio weights sum to 1 and are valid.
    
    Args:
        weights: Dictionary of asset weights
        
    Returns:
        bool: True if valid, False otherwise
    """
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

def format_money(value: float, currency: str = '$') -> str:
    """
    Format monetary values with proper separators and currency symbol.
    
    Args:
        value: Monetary value to format
        currency: Currency symbol to use
        
    Returns:
        str: Formatted monetary string
    """
    try:
        # Handle negative values by placing minus sign before currency
        if value < 0:
            return f"-{currency}{abs(value):,.2f}"
        return f"{currency}{value:,.2f}"
    except Exception as e:
        logger.error(f"Error formatting monetary value: {str(e)}")
        return f"{currency}0.00"


def calculate_moving_average(data: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Calculate simple moving average of a data series.
    
    Args:
        data: Array of price/value data
        window: Moving average window size
        
    Returns:
        np.ndarray: Calculated moving average
    """
    try:
        if len(data) < window:
            return np.array([])
        return np.convolve(data, np.ones(window)/window, mode='valid')
    except Exception as e:
        logger.error(f"Error calculating moving average: {str(e)}")
        return np.array([])

def calculate_ema(data: np.ndarray, span: int = 20) -> np.ndarray:
    """
    Calculate exponential moving average of a data series.
    
    Args:
        data: Array of price/value data
        span: EMA span period
        
    Returns:
        np.ndarray: Calculated EMA
    """
    try:
        if len(data) < span:
            return np.array([])
        alpha = 2.0 / (span + 1)
        return pd.Series(data).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    except Exception as e:
        logger.error(f"Error calculating EMA: {str(e)}")
        return np.array([])

def calculate_correlation(series1: np.ndarray, series2: np.ndarray) -> float:
    """
    Calculate correlation coefficient between two series.
    
    Args:
        series1: First data series
        series2: Second data series
        
    Returns:
        float: Correlation coefficient
    """
    try:
        if len(series1) != len(series2) or len(series1) < 2:
            return 0.0
        return float(np.corrcoef(series1, series2)[0, 1])
    except Exception as e:
        logger.error(f"Error calculating correlation: {str(e)}")
        return 0.0

def normalize_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize data using specified method.
    
    Args:
        data: Array of data to normalize
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        np.ndarray: Normalized data
    """
    try:
        if len(data) < 2:
            return np.array([])
            
        if method == 'minmax':
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val - min_val < 1e-8:
                return np.zeros_like(data)
            return (data - min_val) / (max_val - min_val)
            
        elif method == 'zscore':
            std = np.std(data, ddof=1)
            if std < 1e-8:
                return np.zeros_like(data)
            return (data - np.mean(data)) / std
            
        else:
            logger.error(f"Unsupported normalization method: {method}")
            return np.array([])
            
    except Exception as e:
        logger.error(f"Error normalizing data: {str(e)}")
        return np.array([])
# Trading Constants
def calculate_bollinger_bands(data, window=20, num_std=2.0):
    """
    Calculate Bollinger Bands for a price series.
    
    Args:
        data: Array of price data
        window: Moving average window size
        num_std: Number of standard deviations for bands
        
    Returns:
        tuple: (middle band, upper band, lower band) as numpy arrays
    """
    try:
        if len(data) < window:
            return np.array([]), np.array([]), np.array([])
            
        middle_band = calculate_moving_average(data, window)
        rolling_std = pd.Series(data).rolling(window=window).std().to_numpy()[window-1:]
        
        upper_band = middle_band + (rolling_std * num_std)
        lower_band = middle_band - (rolling_std * num_std)
        
        return middle_band, upper_band, lower_band
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        return np.array([]), np.array([]), np.array([])

def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        data: Array of price data
        period: RSI period
        
    Returns:
        numpy.ndarray: RSI values
    """
    try:
        if len(data) < period + 1:
            return np.array([])
            
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = pd.Series(gains).rolling(window=period).mean().to_numpy()
        avg_loss = pd.Series(losses).rolling(window=period).mean().to_numpy()
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi[period:]
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        return np.array([])

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        data: Array of price data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        tuple: (MACD line, signal line) as numpy arrays
    """
    try:
        if len(data) < slow_period + signal_period:
            return np.array([]), np.array([])
            
        fast_ema = calculate_ema(data, span=fast_period)
        slow_ema = calculate_ema(data, span=slow_period)
        
        macd_line = fast_ema - slow_ema
        signal_line = calculate_ema(macd_line, span=signal_period)
        
        if len(macd_line) == len(signal_line):
            return macd_line, signal_line
        return np.array([]), np.array([])
        

def format_date(date: Union[str, datetime], format_str: str = "%Y-%m-%d") -> str:
    """
    Format date string or datetime object to specified format.
    
    Args:
        date: Date string or datetime object
        format_str: Output date format string
        
    Returns:
        str: Formatted date string
    """
    try:
        if isinstance(date, str):
            date = pd.to_datetime(date)
        return date.strftime(format_str)
    except Exception as e:
        logger.error(f"Error formatting date: {str(e)}")
        return ""

def is_market_hours(timestamp: Union[str, datetime], market_open: str = "09:30", market_close: str = "16:00") -> bool:
    """
    Check if given timestamp is during market hours.
    
    Args:
        timestamp: Timestamp to check
        market_open: Market opening time (HH:MM)
        market_close: Market closing time (HH:MM)
        
    Returns:
        bool: True if timestamp is during market hours
    """
    try:
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
            
        # Convert to datetime.time for comparison
        current_time = timestamp.time()
        open_time = datetime.strptime(market_open, "%H:%M").time()
        close_time = datetime.strptime(market_close, "%H:%M").time()
        
        return open_time <= current_time <= close_time
    except Exception as e:
        logger.error(f"Error checking market hours: {str(e)}")
        return False

def trading_days_between(start_date: Union[str, datetime], end_date: Union[str, datetime]) -> int:
    """
    Calculate number of trading days between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        int: Number of trading days
    """
    try:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Create date range and exclude weekends
        date_range = pd.date_range(start_date, end_date)
        trading_days = len(date_range[date_range.dayofweek < 5])
        
        return trading_days
    except Exception as e:
        logger.error(f"Error calculating trading days: {str(e)}")
        return 0

def round_price(price: float, precision: int = PRICE_PRECISION) -> float:
    """
    Round price to specified precision.
    
    Args:
        price: Price to round
        precision: Decimal places for rounding
        
    Returns:
        float: Rounded price
    """
    try:
        return round(float(price), precision)
    except Exception as e:
        logger.error(f"Error rounding price: {str(e)}")
        return 0.0
        return macd_line, signal_line
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        return np.array([]), np.array([])

# Position and Risk Management Constants
MAX_POSITION_SIZE = 1.0
MIN_POSITION_SIZE = -1.0
MAX_LEVERAGE = 2.0
MIN_TRADE_SIZE = 0.01
DEFAULT_STOP_LOSS = 0.02
DEFAULT_TAKE_PROFIT = 0.05

# Market Parameters
TRADING_DAYS_PER_YEAR = 252  # Standard number of trading days in a year
MIN_DATA_POINTS = 252  # Minimum data points for reliable statistics
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
MAX_TRADES_PER_DAY = 10
CORRELATION_THRESHOLD = 0.7  # Threshold for significant correlation

# Calculation Parameters
ANNUALIZATION_FACTOR = np.sqrt(TRADING_DAYS_PER_YEAR)
PRICE_PRECISION = 2
POSITION_PRECISION = 4
