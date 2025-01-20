import numpy as np
import logging
from typing import Optional, List, Dict, Union

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MetricsCalculator:

    @staticmethod
    def calculate_returns(portfolio_history: List[float]) -> np.ndarray:
        """Calculate returns from portfolio history."""
        # if len(portfolio_history) <= 1:
        #     logger.debug("Insufficient data points for returns calculation")
        #     return np.array([])

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

            return returns

        except Exception as e:
            logger.exception("Error calculating returns")
            return np.array([])

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

        # if len(returns) <= 252:
        #     logger.debug(
        #         f"Insufficient data points for reliable Sortino ratio: {len(returns)}"
        #     )
        #     return 0.0

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

        if len(returns) <= 252:
            logger.debug(
                f"Insufficient data points for reliable Information ratio: {len(returns)}"
            )
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

            if not np.isfinite(avg_excess_return) or not np.isfinite(
                    tracking_error):
                return 0.0

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
        # if len(portfolio_history) <= 1:
        #     logger.debug("Insufficient data points for drawdown calculation")
        #     return 0.0

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

            # logger.debug(f"Calculated maximum drawdown: {max_dd:.3f}")
            return max_dd

        except Exception as e:
            logger.exception("Error calculating maximum drawdown")
            return 0.0
