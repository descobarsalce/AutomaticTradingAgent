"""
Trading platform market utility functions.
Provides market analysis, calculations, and trading-related functions.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Tuple, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def calculate_returns(values: np.ndarray, round_precision: Optional[int] = None) -> np.ndarray:
    """Calculate returns from a series of values."""
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

def calculate_volatility(returns: np.ndarray, annualize: bool = True) -> float:
    """Calculate the volatility of returns."""
    try:
        if len(returns) < 2:
            return 0.0
        vol = np.std(returns, ddof=1)
        if annualize:
            vol = vol * np.sqrt(252)  # Standard trading days in a year
        return float(vol)
    except Exception as e:
        logger.error(f"Error calculating volatility: {str(e)}")
        return 0.0

def calculate_beta(returns: np.ndarray, market_returns: np.ndarray) -> float:
    """Calculate the beta of returns against market returns."""
    try:
        if len(returns) != len(market_returns) or len(returns) < 2:
            return 0.0
        covariance = np.cov(returns, market_returns)[0][1]
        market_variance = np.var(market_returns, ddof=1)
        return float(covariance / market_variance if market_variance != 0 else 0.0)
    except Exception as e:
        logger.error(f"Error calculating beta: {str(e)}")
        return 0.0

def calculate_bollinger_bands(data: np.ndarray, window: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Bollinger Bands for a price series."""
    try:
        if len(data) < window:
            return np.array([]), np.array([]), np.array([])
            
        def calculate_sma(data: np.ndarray, window: int) -> np.ndarray:
            return np.convolve(data, np.ones(window)/window, mode='valid')
            
        middle_band = calculate_sma(data, window)
        rolling_std = pd.Series(data).rolling(window=window).std().to_numpy()[window-1:]
        
        upper_band = middle_band + (rolling_std * num_std)
        lower_band = middle_band - (rolling_std * num_std)
        
        return middle_band, upper_band, lower_band
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        return np.array([]), np.array([]), np.array([])

def calculate_rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index (RSI)."""
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

def calculate_macd(data: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Moving Average Convergence Divergence (MACD)."""
    try:
        if len(data) < slow_period + signal_period:
            return np.array([]), np.array([])
            
        # Calculate EMAs
        def calculate_ema(data: np.ndarray, span: int) -> np.ndarray:
            return pd.Series(data).ewm(span=span, adjust=False).mean().to_numpy()
            
        fast_ema = calculate_ema(data, span=fast_period)
        slow_ema = calculate_ema(data, span=slow_period)
        
        macd_line = fast_ema - slow_ema
        signal_line = calculate_ema(macd_line, span=signal_period)
        
        # Ensure both arrays are the same length
        min_length = min(len(macd_line), len(signal_line))
        macd_line = macd_line[-min_length:]
        signal_line = signal_line[-min_length:]
        
        return macd_line, signal_line
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        return np.array([]), np.array([])

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
