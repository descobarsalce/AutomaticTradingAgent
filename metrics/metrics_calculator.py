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

    @staticmethod
    def calculate_bollinger_bands(
            data: np.ndarray,
            window: int = 20,
            num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands for a price series."""
        try:
            if len(data) < window:
                return np.array([]), np.array([]), np.array([])

            # Calculate SMA
            middle_band = pd.Series(data).rolling(
                window=window).mean().to_numpy()[window - 1:]
            rolling_std = pd.Series(data).rolling(
                window=window).std().to_numpy()[window - 1:]

            upper_band = middle_band + (rolling_std * num_std)
            lower_band = middle_band - (rolling_std * num_std)

            return middle_band, upper_band, lower_band

        except Exception as e:
            logger.exception("Error calculating Bollinger Bands")
            return np.array([]), np.array([]), np.array([])

    @staticmethod
    def calculate_rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index (RSI)."""
        try:
            if len(data) < period + 1:
                return np.array([])

            deltas = np.diff(data)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = pd.Series(gains).rolling(
                window=period).mean().to_numpy()
            avg_loss = pd.Series(losses).rolling(
                window=period).mean().to_numpy()

            rs = np.divide(avg_gain,
                           avg_loss,
                           out=np.zeros_like(avg_gain),
                           where=avg_loss != 0)
            rsi = 100 - (100 / (1 + rs))

            return rsi[period:]

        except Exception as e:
            logger.exception("Error calculating RSI")
            return np.array([])

    @staticmethod
    def calculate_macd(
            data: np.ndarray,
            fast_period: int = 12,
            slow_period: int = 26,
            signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Moving Average Convergence Divergence (MACD)."""
        try:
            if len(data) < slow_period + signal_period:
                return np.array([]), np.array([])

            def calculate_ema(data: np.ndarray, span: int) -> np.ndarray:
                return pd.Series(data).ewm(span=span,
                                           adjust=False).mean().to_numpy()

            fast_ema = calculate_ema(data, span=fast_period)
            slow_ema = calculate_ema(data, span=slow_period)

            macd_line = fast_ema - slow_ema
            signal_line = calculate_ema(macd_line, span=signal_period)

            # Ensure both arrays are the same length
            min_length = min(len(macd_line), len(signal_line))
            macd_line = macd_line[-min_length:]
            signal_line = signal_line[-min_length:]

            return macd_line, signal_line

        except Exception as e:
            logger.exception("Error calculating MACD")
            return np.array([]), np.array([])