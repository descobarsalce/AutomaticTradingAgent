"""
Market data feature source providing OHLCV-based features.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from data.feature_engineering.sources.base_source import BaseFeatureSource

logger = logging.getLogger(__name__)


class MarketDataSource(BaseFeatureSource):
    """Provides OHLCV-based features.

    Features include price changes, returns, momentum, and volume metrics.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize market data source.

        Args:
            config: Optional configuration with:
                - windows: List of rolling window sizes (default: [5, 10, 20])
                - lags: Number of lagged features (default: 5)
        """
        super().__init__(config)
        self.windows = self.config.get('windows', [5, 10, 20])
        self.lags = self.config.get('lags', 5)
        self.priority = 10  # Higher priority for basic features

    def get_available_features(self) -> List[str]:
        """Return list of features this source can provide."""
        base_features = [
            'price_change',
            'returns',
            'log_returns',
            'high_low_ratio',
            'close_open_ratio',
            'typical_price',
            'price_range',
            'gap',
            'volume_change',
            'volume_ma_ratio',
        ]

        # Add lagged features
        lag_features = [f'close_lag{i}' for i in range(1, self.lags + 1)]

        # Add rolling features
        rolling_features = []
        for w in self.windows:
            rolling_features.extend([
                f'rolling_mean_{w}',
                f'rolling_std_{w}',
                f'rolling_min_{w}',
                f'rolling_max_{w}',
            ])

        return base_features + lag_features + rolling_features

    @property
    def dependencies(self) -> List[str]:
        """Required data columns."""
        return ['Close', 'Open', 'High', 'Low', 'Volume']

    def compute_features(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute market data features.

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
            open_col = f'Open_{symbol}'
            high_col = f'High_{symbol}'
            low_col = f'Low_{symbol}'
            volume_col = f'Volume_{symbol}'

            # Skip if required columns missing
            if close_col not in data.columns:
                logger.warning(f"Missing Close column for {symbol}")
                continue

            close = data[close_col]
            has_ohlc = all(
                c in data.columns for c in [open_col, high_col, low_col]
            )
            has_volume = volume_col in data.columns

            for feature in features:
                col_name = f"{feature}_{symbol}"

                try:
                    if feature == 'price_change':
                        result[col_name] = close.diff()

                    elif feature == 'returns':
                        result[col_name] = close.pct_change()

                    elif feature == 'log_returns':
                        result[col_name] = np.log(close / close.shift(1))

                    elif feature == 'high_low_ratio' and has_ohlc:
                        high = data[high_col]
                        low = data[low_col]
                        result[col_name] = high / low.replace(0, np.nan)

                    elif feature == 'close_open_ratio' and has_ohlc:
                        open_price = data[open_col]
                        result[col_name] = close / open_price.replace(0, np.nan)

                    elif feature == 'typical_price' and has_ohlc:
                        high = data[high_col]
                        low = data[low_col]
                        result[col_name] = (high + low + close) / 3

                    elif feature == 'price_range' and has_ohlc:
                        high = data[high_col]
                        low = data[low_col]
                        result[col_name] = high - low

                    elif feature == 'gap' and has_ohlc:
                        open_price = data[open_col]
                        result[col_name] = open_price - close.shift(1)

                    elif feature == 'volume_change' and has_volume:
                        volume = data[volume_col]
                        result[col_name] = volume.pct_change()

                    elif feature == 'volume_ma_ratio' and has_volume:
                        volume = data[volume_col]
                        ma = volume.rolling(window=20).mean()
                        result[col_name] = volume / ma.replace(0, np.nan)

                    elif feature.startswith('close_lag'):
                        lag = int(feature.split('lag')[1])
                        result[col_name] = close.shift(lag)

                    elif feature.startswith('rolling_'):
                        parts = feature.split('_')
                        stat_type = parts[1]  # mean, std, min, max
                        window = int(parts[2])

                        if stat_type == 'mean':
                            result[col_name] = close.rolling(window=window).mean()
                        elif stat_type == 'std':
                            result[col_name] = close.rolling(window=window).std()
                        elif stat_type == 'min':
                            result[col_name] = close.rolling(window=window).min()
                        elif stat_type == 'max':
                            result[col_name] = close.rolling(window=window).max()

                except Exception as e:
                    logger.warning(f"Error computing {feature} for {symbol}: {e}")

        return result
