
"""Market analysis functionality"""
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Tuple, Optional, Union

logger = logging.getLogger(__name__)

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
    """Calculate volatility of returns."""
    try:
        if len(returns) < 2:
            return 0.0
        vol = np.std(returns, ddof=1)
        if annualize:
            vol = vol * np.sqrt(252)
        return float(vol)
    except Exception as e:
        logger.error(f"Error calculating volatility: {str(e)}")
        return 0.0

def calculate_beta(returns: np.ndarray, market_returns: np.ndarray) -> float:
    """Calculate beta against market returns."""
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
    """Calculate Bollinger Bands."""
    try:
        if len(data) < window:
            return np.array([]), np.array([]), np.array([])
        sma = pd.Series(data).rolling(window=window).mean()
        std = pd.Series(data).rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return sma.to_numpy(), upper_band.to_numpy(), lower_band.to_numpy()
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        return np.array([]), np.array([]), np.array([])

def calculate_rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI."""
    try:
        deltas = np.diff(data)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        if down != 0:
            rs = up/down
        else:
            rs = 0
        rsi = np.zeros_like(data)
        rsi[period] = 100. - 100./(1. + rs)
        
        for i in range(period+1, len(data)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
                
            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period
            
            if down != 0:
                rs = up/down
            else:
                rs = 0
            rsi[i] = 100. - 100./(1. + rs)
            
        return rsi
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        return np.array([])

def calculate_macd(data: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate MACD."""
    try:
        exp1 = pd.Series(data).ewm(span=fast_period, adjust=False).mean()
        exp2 = pd.Series(data).ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        return macd.to_numpy(), signal.to_numpy()
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        return np.array([]), np.array([])

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
