"""Metrics calculator using shared technical indicators base class."""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List, Dict, Union, Tuple
from metrics.base_indicators import BaseTechnicalIndicators

logger = logging.getLogger(__name__)

class MetricsCalculator(BaseTechnicalIndicators):
    """Centralized metrics calculation class."""
    
    _cache = {}  # Added cache for heavy computations

    @staticmethod
    def calculate_returns(portfolio_history: List[float],
                        round_precision: Optional[int] = None) -> np.ndarray:
        try:
            portfolio_array = np.array(portfolio_history)
            if not np.all(np.isfinite(portfolio_array)):
                logger.warning("Non-finite values found in portfolio history")
                portfolio_array = portfolio_array[np.isfinite(portfolio_array)]

            if len(portfolio_array) <= 1:
                return np.array([])

            denominator = portfolio_array[:-1]
            if np.any(denominator == 0):
                logger.warning("Zero values found in portfolio history")
                return np.array([])

            returns = np.diff(portfolio_array) / denominator
            returns = returns[np.isfinite(returns)]

            if len(returns) > 0:
                mean, std = np.mean(returns), np.std(returns)
                returns = returns[np.abs(returns - mean) <= 5 * std]
                if round_precision is not None:
                    returns = np.round(returns, round_precision)

            if np.isnan(returns).any():
                raise ValueError("NaN detected in returns calculation")

            return returns
        except Exception as e:
            logger.exception("Error calculating returns")
            return np.array([])

    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray) -> float:
        if not isinstance(returns, np.ndarray) or len(returns) <= 1:
            return 0.0

        try:
            avg_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)

            if std_return > 1e-8:
                sharpe = (avg_return / std_return) * np.sqrt(252)
                return float(np.clip(sharpe, -100, 100))
            return 0.0
        except Exception as e:
            logger.exception("Error calculating Sharpe ratio")
            return 0.0

    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray) -> float:
        if not isinstance(returns, np.ndarray) or len(returns) <= 1:
            return 0.0

        try:
            avg_return = np.mean(returns)
            negative_returns = returns[returns < 0]

            if len(negative_returns) == 0:
                return float('inf') if avg_return > 0 else 0.0

            downside_std = np.std(negative_returns, ddof=1)
            if downside_std > 1e-8:
                sortino = (avg_return / downside_std) * np.sqrt(252)
                return float(np.clip(sortino, -100, 100))
            return 0.0
        except Exception as e:
            logger.exception("Error calculating Sortino ratio")
            return 0.0

    @staticmethod
    def calculate_information_ratio(returns: np.ndarray,
                                  benchmark_returns: Optional[np.ndarray] = None) -> float:
        if not isinstance(returns, np.ndarray) or len(returns) <= 1:
            return 0.0

        try:
            if benchmark_returns is None:
                benchmark_returns = np.zeros_like(returns)

            excess_returns = returns[:len(benchmark_returns)] - benchmark_returns[:len(returns)]
            if len(excess_returns) <= 1:
                return 0.0

            avg_excess_return = np.mean(excess_returns)
            tracking_error = np.std(excess_returns, ddof=1)

            if tracking_error > 1e-8:
                ir = avg_excess_return / tracking_error
                return float(np.clip(ir, -100, 100))
            return 0.0
        except Exception as e:
            logger.exception("Error calculating Information ratio")
            return 0.0

    @staticmethod
    def calculate_maximum_drawdown(portfolio_history: List[float]) -> float:
        try:
            key = tuple(portfolio_history)
            if key in MetricsCalculator._cache:
                return MetricsCalculator._cache[key]

            values = np.array([v for v in portfolio_history if isinstance(v, (int, float)) and v >= 0])
            if len(values) <= 1:
                return 0.0

            peak = np.maximum.accumulate(values)
            drawdowns = (peak - values) / peak
            mdd = float(np.nanmax(drawdowns))
            MetricsCalculator._cache[key] = mdd
            return mdd
        except Exception as e:
            logger.exception("Error calculating maximum drawdown")
            return 0.0

    @staticmethod
    def calculate_volatility(returns: np.ndarray) -> float:
        if not isinstance(returns, np.ndarray) or len(returns) == 0:
            return 0.0
        volatility = np.std(returns, ddof=1) * np.sqrt(252)
        return float(np.clip(volatility, 0, 100))
