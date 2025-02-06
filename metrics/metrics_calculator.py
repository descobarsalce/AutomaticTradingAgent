"""Metrics calculator using shared technical indicators base class."""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List, Dict, Union, Tuple
from metrics.base_indicators import BaseTechnicalIndicators

logger = logging.getLogger(__name__)

class MetricsCalculator(BaseTechnicalIndicators):
    """Centralized metrics calculation class."""

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
            values = np.array([v for v in portfolio_history if isinstance(v, (int, float)) and v >= 0])
            if len(values) <= 1:
                return 0.0

            peak = np.maximum.accumulate(values)
            drawdowns = (peak - values) / peak
            return float(np.nanmax(drawdowns))
        except Exception as e:
            logger.exception("Error calculating maximum drawdown")
            return 0.0

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