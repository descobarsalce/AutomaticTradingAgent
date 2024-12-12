
"""Data processing and validation functionality"""
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

def validate_numeric(value: Union[int, float], min_value: Optional[float] = None, 
                    max_value: Optional[float] = None, allow_zero: bool = True) -> bool:
    """Validate numeric values."""
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
    """Validate DataFrame structure."""
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
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
    """Validate portfolio weights."""
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

def normalize_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """Normalize data."""
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
