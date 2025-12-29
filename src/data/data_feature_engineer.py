"""
Feature engineering pipeline with consolidated technical indicators.

DEPRECATED: This module is deprecated. Please use the new modular feature
engineering system instead:

    from src.data.feature_engineering import FeatureEngineer

    engineer = FeatureEngineer()
    engineer.register_default_sources()
    features = engineer.compute_features(data, symbols=['AAPL', 'MSFT'])

The new system provides:
- Plugin-based architecture for adding custom feature sources
- Automatic feature selection
- Caching for improved performance
- Parallel computation
"""
import warnings
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List
from scipy.fftpack import fft
from src.metrics.base_indicators import BaseTechnicalIndicators

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FeatureEngineer(BaseTechnicalIndicators):
    """Feature engineering using base technical indicators.

    DEPRECATED: Use src.data.feature_engineering.FeatureEngineer instead.
    """

    def __init__(self, n_jobs: int = 1):
        warnings.warn(
            "src.data.data_feature_engineer.FeatureEngineer is deprecated. "
            "Use src.data.feature_engineering.FeatureEngineer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.n_jobs = n_jobs

    def calculate_stochastic_oscillator(
        self,
        data: pd.DataFrame,
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
    ) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        high = data['High']
        low = data['Low']
        close = data['Close']

        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()

        denominator = highest_high - lowest_low
        denominator = denominator.replace(0, np.nan)

        stoch_k = 100 * (close - lowest_low) / denominator
        stoch_k = stoch_k.rolling(window=smooth_k).mean()
        stoch_d = stoch_k.rolling(window=smooth_d).mean()

        return pd.DataFrame({
            'Stoch_%K': stoch_k,
            'Stoch_%D': stoch_d,
        }, index=data.index)

    def calculate_on_balance_volume(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        close = data['Close']
        volume = data['Volume']

        obv = pd.Series(0.0, index=close.index)
        price_change = close.diff()

        obv[price_change > 0] = volume[price_change > 0]
        obv[price_change < 0] = -volume[price_change < 0]

        return obv.cumsum()

    @staticmethod
    def normalize_data(data: pd.Series) -> pd.Series:
        min_val = data.min()
        max_val = data.max()
        if max_val > min_val:
            return (data - min_val) / (max_val - min_val)
        else:
            return pd.Series(0, index=data.index)

    @staticmethod
    def add_lagged_features(data: pd.DataFrame,
                            columns: List[str],
                            lags: int = 3) -> pd.DataFrame:
        for col in columns:
            if col not in data.columns:
                logger.warning(f"Column '{col}' not found for lagged features.")
                continue
            for lag in range(1, lags + 1):
                data[f"{col}_lag{lag}"] = data[col].shift(lag)
        return data

    @staticmethod
    def add_rolling_features(data: pd.DataFrame, column: str,
                             windows: List[int]) -> pd.DataFrame:
        if column not in data.columns:
            logger.warning(f"Column '{column}' not found for rolling features.")
            return data

        for w in windows:
            data[f"{column}_rolling_mean_{w}"] = data[column].rolling(window=w).mean()
            data[f"{column}_rolling_std_{w}"] = data[column].rolling(window=w).std()
            data[f"{column}_rolling_min_{w}"] = data[column].rolling(window=w).min()
            data[f"{column}_rolling_max_{w}"] = data[column].rolling(window=w).max()
        return data

    @staticmethod
    def add_fourier_transform(prices: pd.Series,
                              top_n: int = 3,
                              window: int = 30) -> pd.DataFrame:
        if len(prices) < window:
            logger.warning(f"Data length {len(prices)} < window size {window}.")
            return pd.DataFrame(index=prices.index)

        result = pd.DataFrame(index=prices.index)
        try:
            for i in range(window, len(prices) + 1):
                window_data = prices.iloc[i - window:i]
                fft_result = fft(window_data.values)
                for j in range(top_n):
                    result.loc[prices.index[i - 1], f'FFT_{j + 1}'] = np.abs(fft_result[j])
            return result.ffill()
        except Exception as e:
            logger.error(f"FFT calculation error: {e}")
            return pd.DataFrame(index=prices.index)

    def prepare_data(self, portfolio_data: pd.DataFrame, current_index: Optional[int] = None) -> pd.DataFrame:
        """Main pipeline to prepare features from portfolio data."""
        if current_index is not None:
            portfolio_data = portfolio_data.iloc[:current_index + 1].copy()
        if not isinstance(portfolio_data, pd.DataFrame) or portfolio_data.empty:
            raise ValueError("Invalid portfolio data")

        symbols = sorted(list(set(col.split('_')[1] for col in portfolio_data.columns if '_' in col)))
        prepared_data = portfolio_data.copy()

        for symbol in symbols:
            try:
                close_col = f'Close_{symbol}'
                high_col = f'High_{symbol}'
                low_col = f'Low_{symbol}'
                volume_col = f'Volume_{symbol}'

                if close_col in portfolio_data.columns:
                    # Use parent class methods for technical indicators
                    prepared_data[f'RSI_{symbol}'] = super().calculate_rsi(portfolio_data[close_col])
                    prepared_data[f'Volatility_{symbol}'] = super().calculate_volatility(portfolio_data[close_col])

                    # Bollinger Bands
                    sma, upper, lower = super().calculate_bollinger_bands(portfolio_data[close_col])
                    prepared_data[f'BB_Upper_{symbol}'] = upper
                    prepared_data[f'BB_Lower_{symbol}'] = lower
                    prepared_data[f'BB_SMA_{symbol}'] = sma

                    # MACD
                    macd, signal = super().calculate_macd(portfolio_data[close_col])
                    prepared_data[f'MACD_{symbol}'] = macd
                    prepared_data[f'MACD_Signal_{symbol}'] = signal

                    # Add additional features
                    self.add_lagged_features(prepared_data, [close_col], lags=5)
                    self.add_rolling_features(prepared_data, close_col, [7, 14, 30])

                    fft_features = self.add_fourier_transform(portfolio_data[close_col])
                    if not fft_features.empty:
                        for col in fft_features.columns:
                            prepared_data[f'{col}_{symbol}'] = fft_features[col]

                # Stochastic oscillator if we have high/low data
                if all(col in portfolio_data.columns for col in [high_col, low_col, close_col]):
                    symbol_data = pd.DataFrame({
                        'High': portfolio_data[high_col],
                        'Low': portfolio_data[low_col],
                        'Close': portfolio_data[close_col]
                    })
                    stoch = self.calculate_stochastic_oscillator(symbol_data)
                    prepared_data[f'Stoch_K_{symbol}'] = stoch['Stoch_%K']
                    prepared_data[f'Stoch_D_{symbol}'] = stoch['Stoch_%D']

                # Volume-based indicators
                if volume_col in portfolio_data.columns:
                    symbol_data = pd.DataFrame({
                        'Close': portfolio_data[close_col],
                        'Volume': portfolio_data[volume_col]
                    })
                    prepared_data[f'OBV_{symbol}'] = self.calculate_on_balance_volume(symbol_data)

            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {str(e)}")

        prepared_data.dropna(inplace=True)
        return prepared_data
