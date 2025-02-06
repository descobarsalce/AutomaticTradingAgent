"""
Metrics calculator using shared technical indicators base class.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List, Dict, Union, Tuple
from metrics.base_indicators import BaseTechnicalIndicators

logger = logging.getLogger(__name__)

class MetricsCalculator(BaseTechnicalIndicators):
    """
    Centralized metrics calculation class.
    Inherits from BaseTechnicalIndicators for shared technical analysis.
    """

    @staticmethod
    def calculate_returns(portfolio_history: List[float],
                        round_precision: Optional[int] = None) -> np.ndarray:
        """Calculate returns from portfolio history."""
        try:
            portfolio_array = np.array(portfolio_history)
            if not np.all(np.isfinite(portfolio_array)):
                logger.warning("Non-finite values found in portfolio history")
                portfolio_array = portfolio_array[np.isfinite(portfolio_array)]

            if len(portfolio_array) <= 1:
                logger.debug("Insufficient valid data points after filtering")
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

            return returns

        except Exception as e:
            logger.exception("Error calculating returns")
            return np.array([])

    @staticmethod
    def calculate_volatility(returns: np.ndarray,
                             annualize: bool = True) -> float:
        """Calculate return volatility with optional annualization."""
        if not isinstance(returns, np.ndarray):
            logger.warning("Invalid input type for volatility calculation")
            return 0.0

        if len(returns) <= 1:
            logger.debug("Insufficient data points for volatility calculation")
            return 0.0

        try:
            valid_returns = returns[np.isfinite(returns)]
            if len(valid_returns) <= 1:
                return 0.0

            vol = np.std(valid_returns, ddof=1)
            if annualize:
                vol = vol * np.sqrt(252)  # Standard trading days in a year
            return float(vol)

        except Exception as e:
            logger.exception("Error calculating volatility")
            return 0.0

    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray) -> float:
        """Calculate Sharpe ratio from returns."""
        if not isinstance(returns, np.ndarray):
            logger.warning("Invalid input type for returns calculation")
            return 0.0

        try:
            valid_returns = returns[np.isfinite(returns)]
            if len(valid_returns) <= 1:
                return 0.0

            avg_return = np.mean(valid_returns)
            std_return = np.std(valid_returns, ddof=1)

            if not np.isfinite(avg_return) or not np.isfinite(std_return):
                return 0.0

            if std_return > 1e-8:
                annualization_factor = np.sqrt(252)
                sharpe = (avg_return / std_return) * annualization_factor
                return float(np.clip(sharpe, -100, 100))

            return 0.0

        except Exception as e:
            logger.exception("Error calculating Sharpe ratio")
            return 0.0

    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray) -> float:
        """Calculate Sortino ratio from returns."""
        if not isinstance(returns, np.ndarray):
            logger.warning("Invalid input type for Sortino ratio calculation")
            return 0.0

        try:
            valid_returns = returns[np.isfinite(returns)]
            if len(valid_returns) <= 1:
                return 0.0

            avg_return = np.mean(valid_returns)
            negative_returns = valid_returns[valid_returns < 0]

            if len(negative_returns) == 0:
                return float('inf') if avg_return > 0 else 0.0

            downside_std = np.std(negative_returns, ddof=1)

            if not np.isfinite(avg_return) or not np.isfinite(downside_std):
                return 0.0

            if downside_std > 1e-8:
                annualization_factor = np.sqrt(252)
                sortino = (avg_return / downside_std) * annualization_factor
                return float(np.clip(sortino, -100, 100))

            return 0.0

        except Exception as e:
            logger.exception("Error calculating Sortino ratio")
            return 0.0

    @staticmethod
    def calculate_information_ratio(
            returns: np.ndarray,
            benchmark_returns: Optional[np.ndarray] = None) -> float:
        """Calculate Information ratio from returns."""
        if not isinstance(returns, np.ndarray):
            logger.warning(
                "Invalid input type for Information ratio calculation")
            return 0.0

        try:
            if benchmark_returns is None:
                benchmark_returns = np.zeros_like(returns)

            min_length = min(len(returns), len(benchmark_returns))
            returns = returns[:min_length]
            benchmark_returns = benchmark_returns[:min_length]

            excess_returns = returns - benchmark_returns
            valid_returns = excess_returns[np.isfinite(excess_returns)]

            if len(valid_returns) <= 1:
                return 0.0

            avg_excess_return = np.mean(valid_returns)
            tracking_error = np.std(valid_returns, ddof=1)

            if tracking_error > 1e-8:
                information_ratio = avg_excess_return / tracking_error
                return float(np.clip(information_ratio, -100, 100))

            return 0.0

        except Exception as e:
            logger.exception("Error calculating Information ratio")
            return 0.0

    @staticmethod
    def calculate_maximum_drawdown(portfolio_history: List[float]) -> float:
        """Calculate maximum drawdown from portfolio history."""
        try:
            values = np.array([
                v for v in portfolio_history
                if isinstance(v, (int, float)) and v >= 0
            ])
            if len(values) <= 1:
                logger.warning(
                    "No valid values for drawdown calculation after filtering")
                return 0.0

            peak = np.maximum.accumulate(values)
            drawdowns = (peak - values) / peak
            max_dd = float(np.nanmax(drawdowns))
            return max_dd

        except Exception as e:
            logger.exception("Error calculating maximum drawdown")
            return 0.0

    def calculate_bollinger_bands(
            self,
            data: np.ndarray,
            window: int = 20,
            num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands for a price series."""
        try:
            if len(data) < window:
                return np.array([]), np.array([]), np.array([])
            return super().calculate_bollinger_bands(data, window, num_std)
        except Exception as e:
            logger.exception("Error calculating Bollinger Bands")
            return np.array([]), np.array([]), np.array([])

    def calculate_rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index (RSI)."""
        try:
            if len(data) < period + 1:
                return np.array([])
            return super().calculate_rsi(data, period)
        except Exception as e:
            logger.exception("Error calculating RSI")
            return np.array([])

    def calculate_macd(
            self,
            data: np.ndarray,
            fast_period: int = 12,
            slow_period: int = 26,
            signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Moving Average Convergence Divergence (MACD)."""
        try:
            if len(data) < slow_period + signal_period:
                return np.array([]), np.array([])
            return super().calculate_macd(data, fast_period, slow_period, signal_period)
        except Exception as e:
            logger.exception("Error calculating MACD")
            return np.array([]), np.array([])
def calculate_out_of_sample_metrics(in_sample_results, out_of_sample_results):
    """
    Compare in-sample vs out-of-sample performance to detect overfitting.
    Returns a dict with performance degradation metrics.
    """
    metrics = {}
    
    # Calculate performance degradation
    metrics['sharpe_ratio_degradation'] = (
        out_of_sample_results['sharpe_ratio'] / in_sample_results['sharpe_ratio']
    )
    
    metrics['returns_degradation'] = (
        out_of_sample_results['returns'] / in_sample_results['returns']
    )
    
    # Profit factor comparison
    metrics['profit_factor_degradation'] = (
        out_of_sample_results['profit_factor'] / in_sample_results['profit_factor']
    )
    
    return metrics
def perform_monte_carlo_analysis(returns, n_simulations=1000, confidence_level=0.95):
    """
    Perform Monte Carlo simulation to estimate strategy robustness.
    """
    results = []
    for _ in range(n_simulations):
        # Resample returns with replacement
        simulated_returns = np.random.choice(returns, size=len(returns), replace=True)
        cumulative_returns = (1 + simulated_returns).cumprod()
        results.append(cumulative_returns[-1])
    
    # Calculate confidence intervals
    conf_interval = np.percentile(results, [(1-confidence_level)/2, 1-(1-confidence_level)/2])
    
    return {
        'mean_terminal_value': np.mean(results),
        'confidence_interval': conf_interval,
        'worst_case': np.min(results),
        'best_case': np.max(results)
    }
