"""
Trading platform shared utilities module.
Provides organized utility functions and constants used across the platform.
"""

# Import data validation utilities
from .data_utils import (
    validate_numeric,
    validate_dataframe,
    validate_portfolio_weights,
    normalize_data
)

# Import market analysis utilities
from .market_utils import (
    calculate_returns,
    calculate_volatility,
    calculate_beta,
    calculate_bollinger_bands,
    calculate_rsi,
    calculate_macd,
    is_market_hours,
    trading_days_between
)

# Import formatting utilities
from .formatting_utils import (
    format_timestamp,
    format_money,
    format_date,
    round_price
)

# Import common trading parameters and constants
from .common import (
    # Position and Risk Management
    MAX_POSITION_SIZE,
    MIN_POSITION_SIZE,
    MAX_LEVERAGE,
    MIN_TRADE_SIZE,
    DEFAULT_STOP_LOSS,
    DEFAULT_TAKE_PROFIT,
    
    # Market Parameters
    TRADING_DAYS_PER_YEAR,
    MIN_DATA_POINTS,
    RISK_FREE_RATE,
    MAX_TRADES_PER_DAY,
    CORRELATION_THRESHOLD,
    
    # Precision Settings
    PRICE_PRECISION,
    POSITION_PRECISION,
    
    # Calculation Constants
    ANNUALIZATION_FACTOR,
    
    # Validation Functions
    validate_trading_params
)

__all__ = [
    # Validation functions
    'validate_numeric',
    'validate_dataframe',
    'validate_trading_params',
    'validate_portfolio_weights',
    
    # Calculation functions
    'calculate_returns',
    'calculate_volatility',
    'calculate_beta',
    'normalize_data',
    
    # Formatting functions
    'format_timestamp',
    'format_money',
    'format_date',
    'round_price',
    
    # Trading constants
    'MAX_POSITION_SIZE',
    'MIN_POSITION_SIZE',
    'DEFAULT_STOP_LOSS',
    'DEFAULT_TAKE_PROFIT',
    'RISK_FREE_RATE',
    'TRADING_DAYS_PER_YEAR',
    'MIN_DATA_POINTS',
    'ANNUALIZATION_FACTOR',
    'CORRELATION_THRESHOLD',
    'MAX_TRADES_PER_DAY',
    'MAX_LEVERAGE',
    'MIN_TRADE_SIZE',
    'PRICE_PRECISION',
    'POSITION_PRECISION'
]
