"""
Trading platform shared utilities module.
Contains common constants and validation functions used across the platform.
"""

from typing import Dict, Any, Union, Optional
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

# Position and Risk Management
MAX_POSITION_SIZE = 1.0  # Maximum allowed position size
MIN_POSITION_SIZE = -1.0  # Minimum allowed position size (for short positions)
MAX_LEVERAGE = 2.0  # Maximum allowed leverage
MIN_TRADE_SIZE = 0.01  # Minimum trade size allowed
DEFAULT_STOP_LOSS = 0.02  # Default stop loss percentage (2%)
DEFAULT_TAKE_PROFIT = 0.05  # Default take profit percentage (5%)

# Market Parameters
TRADING_DAYS_PER_YEAR = 252  # Standard number of trading days in a year
MIN_DATA_POINTS = 252  # Minimum data points required for reliable statistics
RISK_FREE_RATE = 0.02  # Annual risk-free rate (2%)
MAX_TRADES_PER_DAY = 10  # Maximum number of trades allowed per day
CORRELATION_THRESHOLD = 0.7  # Threshold for significant correlation between assets

# Precision Settings
PRICE_PRECISION = 2  # Decimal places for price values
POSITION_PRECISION = 4  # Decimal places for position sizes

# Calculation Constants
ANNUALIZATION_FACTOR = 252 ** 0.5  # Square root of trading days for annualization