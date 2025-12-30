"""
Technical indicators feature source.

Wraps the BaseTechnicalIndicators for use in the plugin architecture.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from scipy.fftpack import fft
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    fft = None
    logging.getLogger(__name__).warning(
        "scipy not available, FFT features will be disabled"
    )

from src.data.feature_engineering.sources.base_source import BaseFeatureSource
from src.metrics.base_indicators import BaseTechnicalIndicators

logger = logging.getLogger(__name__)


class TechnicalSource(BaseFeatureSource, BaseTechnicalIndicators):
    """Provides technical indicator features.

    Includes RSI, MACD, Bollinger Bands, Stochastic Oscillator,
    On-Balance Volume, volatility, and FFT features.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize technical source.

        Args:
            config: Optional configuration with:
                - rsi_period: RSI period (default: 14)
                - macd_fast: MACD fast period (default: 12)
                - macd_slow: MACD slow period (default: 26)
                - macd_signal: MACD signal period (default: 9)
                - bb_window: Bollinger Bands window (default: 20)
                - bb_std: Bollinger Bands std dev (default: 2.0)
                - volatility_window: Volatility window (default: 20)
                - stoch_period: Stochastic period (default: 14)
                - fft_components: Number of FFT components (default: 3)
                - fft_window: FFT window size (default: 30)
        """
        BaseFeatureSource.__init__(self, config)
        self.rsi_period = self.config.get('rsi_period', 14)
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        self.bb_window = self.config.get('bb_window', 20)
        self.bb_std = self.config.get('bb_std', 2.0)
        self.volatility_window = self.config.get('volatility_window', 20)
        self.stoch_period = self.config.get('stoch_period', 14)
        self.fft_components = self.config.get('fft_components', 3)
        self.fft_window = self.config.get('fft_window', 30)
        self.priority = 8

    def get_available_features(self) -> List[str]:
        """Return list of features this source can provide."""
        features = [
            'rsi',
            'macd',
            'macd_signal',
            'macd_histogram',
            'bb_upper',
            'bb_lower',
            'bb_sma',
            'bb_width',
            'bb_percent',
            'volatility',
            'stoch_k',
            'stoch_d',
            'obv',
            'obv_ema',
        ]

        # Add FFT features
        for i in range(1, self.fft_components + 1):
            features.append(f'fft_{i}')

        return features

    @property
    def dependencies(self) -> List[str]:
        """Required data columns."""
        return ['Close']

    def compute_features(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute technical indicator features.

        Args:
            data: Input DataFrame with OHLCV data
            symbols: List of symbols to compute features for
            feature_names: Optional list of specific features to compute

        Returns:
            DataFrame with computed features
        """
        features = feature_names or self.get_available_features()
        result = pd.DataFrame(index=data.index)

        for symbol in symbols:
            close_col = f'Close_{symbol}'
            high_col = f'High_{symbol}'
            low_col = f'Low_{symbol}'
            volume_col = f'Volume_{symbol}'

            if close_col not in data.columns:
                logger.warning(f"Missing Close column for {symbol}")
                continue

            close = data[close_col]
            has_hl = high_col in data.columns and low_col in data.columns
            has_volume = volume_col in data.columns

            for feature in features:
                col_name = f"{feature}_{symbol}"

                try:
                    if feature == 'rsi':
                        result[col_name] = self.calculate_rsi(
                            close, period=self.rsi_period
                        )

                    elif feature == 'macd':
                        macd, _ = self.calculate_macd(
                            close,
                            fast_period=self.macd_fast,
                            slow_period=self.macd_slow,
                            signal_period=self.macd_signal,
                        )
                        result[col_name] = macd

                    elif feature == 'macd_signal':
                        _, signal = self.calculate_macd(
                            close,
                            fast_period=self.macd_fast,
                            slow_period=self.macd_slow,
                            signal_period=self.macd_signal,
                        )
                        result[col_name] = signal

                    elif feature == 'macd_histogram':
                        macd, signal = self.calculate_macd(
                            close,
                            fast_period=self.macd_fast,
                            slow_period=self.macd_slow,
                            signal_period=self.macd_signal,
                        )
                        result[col_name] = macd - signal

                    elif feature == 'bb_upper':
                        _, upper, _ = self.calculate_bollinger_bands(
                            close, window=self.bb_window, num_std=self.bb_std
                        )
                        result[col_name] = upper

                    elif feature == 'bb_lower':
                        _, _, lower = self.calculate_bollinger_bands(
                            close, window=self.bb_window, num_std=self.bb_std
                        )
                        result[col_name] = lower

                    elif feature == 'bb_sma':
                        sma, _, _ = self.calculate_bollinger_bands(
                            close, window=self.bb_window, num_std=self.bb_std
                        )
                        result[col_name] = sma

                    elif feature == 'bb_width':
                        sma, upper, lower = self.calculate_bollinger_bands(
                            close, window=self.bb_window, num_std=self.bb_std
                        )
                        result[col_name] = (upper - lower) / sma.replace(0, np.nan)

                    elif feature == 'bb_percent':
                        sma, upper, lower = self.calculate_bollinger_bands(
                            close, window=self.bb_window, num_std=self.bb_std
                        )
                        band_width = upper - lower
                        result[col_name] = (close - lower) / band_width.replace(0, np.nan)

                    elif feature == 'volatility':
                        result[col_name] = self.calculate_volatility(
                            close, window=self.volatility_window
                        )

                    elif feature == 'stoch_k' and has_hl:
                        k, _ = self._calculate_stochastic(
                            data[high_col], data[low_col], close
                        )
                        result[col_name] = k

                    elif feature == 'stoch_d' and has_hl:
                        _, d = self._calculate_stochastic(
                            data[high_col], data[low_col], close
                        )
                        result[col_name] = d

                    elif feature == 'obv' and has_volume:
                        result[col_name] = self._calculate_obv(
                            close, data[volume_col]
                        )

                    elif feature == 'obv_ema' and has_volume:
                        obv = self._calculate_obv(close, data[volume_col])
                        result[col_name] = obv.ewm(span=20, adjust=False).mean()

                    elif feature.startswith('fft_'):
                        fft_num = int(feature.split('_')[1])
                        fft_features = self._calculate_fft(close, fft_num)
                        if fft_features is not None:
                            result[col_name] = fft_features

                except Exception as e:
                    logger.warning(f"Error computing {feature} for {symbol}: {e}")

        return result

    def _calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
    ) -> tuple:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()

        denominator = highest_high - lowest_low
        denominator = denominator.replace(0, np.nan)

        stoch_k = 100 * (close - lowest_low) / denominator
        stoch_k = stoch_k.rolling(window=smooth_k).mean()
        stoch_d = stoch_k.rolling(window=smooth_d).mean()

        return stoch_k, stoch_d

    def _calculate_obv(
        self,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = pd.Series(0.0, index=close.index)
        price_change = close.diff()

        obv[price_change > 0] = volume[price_change > 0]
        obv[price_change < 0] = -volume[price_change < 0]

        return obv.cumsum()

    def _calculate_fft(
        self,
        prices: pd.Series,
        component: int,
    ) -> Optional[pd.Series]:
        """Calculate FFT component using rolling window."""
        if not HAS_SCIPY or fft is None:
            logger.debug("FFT not available - scipy not installed")
            return None

        if len(prices) < self.fft_window:
            return None

        result = pd.Series(index=prices.index, dtype=float)

        try:
            for i in range(self.fft_window, len(prices) + 1):
                window_data = prices.iloc[i - self.fft_window:i]
                fft_result = fft(window_data.values)
                if component <= len(fft_result):
                    result.iloc[i - 1] = np.abs(fft_result[component - 1])

            return result.ffill()
        except Exception as e:
            logger.warning(f"FFT calculation error: {e}")
            return None
