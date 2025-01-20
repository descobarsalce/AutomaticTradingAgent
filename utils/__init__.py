"""
Trading platform shared utilities module.
Provides organized utility functions and constants used across the platform.
"""

from typing import Dict, Any, Union, Optional
import logging

# Configure root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from .common import (
    validate_numeric,
    validate_dataframe,
    validate_portfolio_weights,
    validate_trading_params,
    format_timestamp,
    format_money,
    format_date,
    round_price,
    is_market_hours,
    trading_days_between,
    type_check,
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
    ANNUALIZATION_FACTOR
)

from .callbacks import (
    PortfolioMetricsCallback,
    ProgressBarCallback
)

__all__ = [
    'validate_numeric',
    'validate_dataframe',
    'validate_portfolio_weights',
    'validate_trading_params',
    'format_timestamp',
    'format_money',
    'format_date',
    'round_price',
    'is_market_hours',
    'trading_days_between',
    'type_check',
    'PortfolioMetricsCallback',
    'ProgressBarCallback',
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
    'ANNUALIZATION_FACTOR'
]