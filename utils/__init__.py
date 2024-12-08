"""
Trading platform shared utilities module.
Provides common functions and constants used across the platform.
"""
from .common import (
    # Validation functions
    validate_numeric,
    validate_dataframe,
    validate_trading_params,
    validate_portfolio_weights,
    
    # Calculation functions
    calculate_returns,
    calculate_volatility,
    calculate_beta,
    
    # Formatting functions
    format_timestamp,
    format_money,
    
    # Trading constants
    MAX_POSITION_SIZE,
    MIN_POSITION_SIZE,
    DEFAULT_STOP_LOSS,
    DEFAULT_TAKE_PROFIT,
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    MIN_DATA_POINTS,
    ANNUALIZATION_FACTOR,
    CORRELATION_THRESHOLD,
    MAX_TRADES_PER_DAY,
    MAX_LEVERAGE,
    MIN_TRADE_SIZE,
    PRICE_PRECISION,
    POSITION_PRECISION
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
    
    # Formatting functions
    'format_timestamp',
    'format_money',
    
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
