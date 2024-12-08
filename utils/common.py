"""
Common utility functions shared across the trading platform.
"""
from typing import Union, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

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
        return f"{currency}{value:,.2f}"
    except Exception as e:
        logger.error(f"Error formatting monetary value: {str(e)}")
        return f"{currency}0.00"

# Trading Constants
MAX_POSITION_SIZE = 1.0
MIN_POSITION_SIZE = -1.0
DEFAULT_STOP_LOSS = 0.02
DEFAULT_TAKE_PROFIT = 0.05
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
TRADING_DAYS_PER_YEAR = 252
MAX_LEVERAGE = 2.0
MIN_TRADE_SIZE = 0.01
PRICE_PRECISION = 2
POSITION_PRECISION = 4
