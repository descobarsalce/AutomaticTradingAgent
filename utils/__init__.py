
"""
Trading platform shared utilities module.
Provides organized utility functions and constants used across the platform.
"""

from typing import Dict, Any, Union, Optional
import logging

# Configure root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Import data validation utilities
from .data_utils import (
    validate_numeric,
    validate_dataframe,
    validate_portfolio_weights,
    normalize_data
)

# Import market analysis utilities
from .market_utils import (
    is_market_hours,
    trading_days_between
)

# Import metrics calculator
from metrics.metrics_calculator import MetricsCalculator

# Import formatting utilities
from .formatting_utils import (
    format_timestamp,
    format_money,
    format_date,
    round_price
)

# Import callback utilities
from .callbacks import (
    PortfolioMetricsCallback,
    ProgressBarCallback
)

# Import common trading parameters and constants
from .common import (
    MAX_POSITION_SIZE,
    MIN_POSITION_SIZE,
    MAX_LEVERAGE,
    MIN_TRADE_SIZE,
    DEFAULT_STOP_LOSS,
    DEFAULT_TAKE_PROFIT,
    TRADING_DAYS_PER_YEAR,
    MIN_DATA_POINTS,
    RISK_FREE_RATE,
    MAX_TRADES_PER_DAY,
    CORRELATION_THRESHOLD,
    PRICE_PRECISION,
    POSITION_PRECISION,
    ANNUALIZATION_FACTOR,
    validate_trading_params,
    type_check
)

__all__ = [
    # Validation functions
    'validate_numeric',
    'validate_dataframe', 
    'validate_trading_params',
    'validate_portfolio_weights',
    'normalize_data',
    'type_check',
    
    # Calculator
    'MetricsCalculator',
    
    # Market utilities
    'is_market_hours',
    'trading_days_between',
    
    # Formatting functions 
    'format_timestamp',
    'format_money',
    'format_date',
    'round_price',
    
    # Callbacks
    'PortfolioMetricsCallback',
    'ProgressBarCallback',
    
    # Constants
    'MAX_POSITION_SIZE',
    'MIN_POSITION_SIZE', 
    'MAX_LEVERAGE',
    'MIN_TRADE_SIZE',
    'DEFAULT_STOP_LOSS',
    'DEFAULT_TAKE_PROFIT',
    'TRADING_DAYS_PER_YEAR',
    'MIN_DATA_POINTS',
    'RISK_FREE_RATE',
    'MAX_TRADES_PER_DAY',
    'CORRELATION_THRESHOLD',
    'PRICE_PRECISION',
    'POSITION_PRECISION',
    'ANNUALIZATION_FACTOR',
]
