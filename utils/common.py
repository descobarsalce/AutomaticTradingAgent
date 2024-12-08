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

def calculate_returns(values: np.ndarray) -> np.ndarray:
    """
    Calculate returns from a series of values.
    
    Args:
        values: Array of values
        
    Returns:
        np.ndarray: Calculated returns
    """
    try:
        if len(values) < 2:
            return np.array([])
        return np.diff(values) / values[:-1]
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

# Trading Constants
MAX_POSITION_SIZE = 1.0
MIN_POSITION_SIZE = -1.0
DEFAULT_STOP_LOSS = 0.02
DEFAULT_TAKE_PROFIT = 0.05
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
TRADING_DAYS_PER_YEAR = 252
