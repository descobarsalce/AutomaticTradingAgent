"""
Feature engineering pipeline with consolidated technical indicators.
Removed duplicate validation functions and uses centralized validation from utils.common
"""

import pandas as pd
import numpy as np
import logging
import concurrent.futures
from typing import Dict, Optional, List
from scipy.fftpack import fft
from utils.common import (
    validate_numeric,
    validate_dataframe,
    validate_portfolio_weights
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FeatureEngineer:
    """
    A single-class feature engineering pipeline that includes:
    1. Technical indicators calculation using centralized functions
    2. Fourier Transform features
    3. Correlations among symbols
    4. Lagged and rolling features
    5. Data preparation orchestration
    """

    def __init__(self, n_jobs: int = 1):
        """
        :param n_jobs: Number of threads to use for parallel correlation calculation.
        """
        self.n_jobs = n_jobs

    # Remove duplicate validation methods as they're now in utils.common

    # NOTE THAT THIS NORMALIZATION IS WRONG BECAUSE IT USES DATA FROM AFTER THE TRAIN/TEST PERIOD. IT CANNOT BE FORWARD LOOKING, SO I NEED TO CONFIRM.
    @staticmethod
    def normalize_data(data: pd.Series) -> pd.Series:
        """Normalizes a series to [0,1]. If min==max, returns all zeros."""
        min_val = data.min()
        max_val = data.max()
        if max_val > min_val:
            return (data - min_val) / (max_val - min_val)
        else:
            return pd.Series(0, index=data.index)

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI):
            RSI = 100 - (100 / (1 + RS))
            where RS = (Avg. gain over n) / (Avg. loss over n)
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()

        # Avoid division-by-zero if there's no loss
        loss = loss.replace(0, 1e-10)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.clip(lower=0, upper=100)

    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series,
                                  window: int = 20) -> pd.DataFrame:
        """
        Bollinger Bands:
            Middle Band = SMA over 'window'
            Upper Band = SMA + 2 * STD
            Lower Band = SMA - 2 * STD
        """
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        return pd.DataFrame({
            'Upper_Band': sma + (2 * std),
            'Lower_Band': sma - (2 * std),
            'SMA': sma
        })

    @staticmethod
    def calculate_macd(prices: pd.Series,
                       short_window: int = 12,
                       long_window: int = 26,
                       signal_window: int = 9) -> pd.DataFrame:
        """
        MACD = EMA_short - EMA_long
        Signal = EMA(MACD, signal_window)
        """
        ema_short = prices.ewm(span=short_window, adjust=False).mean()
        ema_long = prices.ewm(span=long_window, adjust=False).mean()
        macd = ema_short - ema_long
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return pd.DataFrame({'MACD': macd, 'Signal': signal})

    @staticmethod
    def calculate_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
        """
        Simple rolling std of percentage change.
        """
        return prices.pct_change().rolling(window=window).std()

    @staticmethod
    def calculate_stochastic_oscillator(data: pd.DataFrame,
                                        k_window: int = 14,
                                        d_window: int = 3) -> pd.DataFrame:
        """
        Stochastic Oscillator:
            %K = (Close - LowestLow) / (HighestHigh - LowestLow) * 100
            %D = SMA(%K, d_window)

        Expects data with columns ['High','Low','Close'].
        """
        if not all(col in data.columns for col in ['High', 'Low', 'Close']):
            raise ValueError(
                "Data must contain High, Low, Close for Stochastic.")

        lowest_low = data['Low'].rolling(window=k_window).min()
        highest_high = data['High'].rolling(window=k_window).max()

        # if highest_high - lowest_low == 0 => fill with np.nan
        denom = (highest_high - lowest_low).replace(0, np.nan)
        k_value = ((data['Close'] - lowest_low) / denom) * 100
        d_value = k_value.rolling(window=d_window).mean()

        return pd.DataFrame(
            {
                'Stoch_%K': k_value.clip(0, 100),
                'Stoch_%D': d_value.clip(0, 100)
            },
            index=data.index)

    @staticmethod
    def calculate_money_flow_index(data: pd.DataFrame,
                                   period: int = 14) -> pd.Series:
        """
        Money Flow Index (MFI):
            Typical Price = (High + Low + Close)/3
            Money Flow = Typical Price * Volume
            MFI = 100 - (100 / (1 + pos_flow / neg_flow)) over 'period'

        Expects: ['High','Low','Close','Volume'] 
        """
        for col in ['High', 'Low', 'Close', 'Volume']:
            if col not in data.columns:
                raise ValueError(
                    "Data must contain High, Low, Close, Volume for MFI.")

        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']

        direction = typical_price.diff()
        pos_flow = money_flow.where(direction > 0, 0).rolling(period).sum()
        neg_flow = money_flow.where(direction < 0, 0).rolling(period).sum()

        # Avoid zero division
        pos_flow = pos_flow.replace(0, 1e-10)
        neg_flow = neg_flow.replace(0, 1e-10)

        mf_ratio = pos_flow / neg_flow
        mfi = 100 - (100 / (1 + mf_ratio))
        return mfi.clip(lower=0, upper=100)

    @staticmethod
    def calculate_on_balance_volume(data: pd.DataFrame) -> pd.Series:
        """
        OBV:
            OBV_t = OBV_{t-1} + { Volume if close>prev_close
                                  -Volume if close<prev_close
                                  0 otherwise }

        Expects: ['Close','Volume'] in DataFrame
        """
        if not all(col in data.columns for col in ['Close', 'Volume']):
            raise ValueError("Data must contain 'Close' and 'Volume' for OBV.")

        obv = [0]
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])

        return pd.Series(obv, index=data.index, name='OBV')

    @staticmethod
    def calculate_accumulation_distribution_line(
            data: pd.DataFrame) -> pd.Series:
        """
        A/D Line:
            CLV = ((Close - Low) - (High - Close)) / (High - Low)
            ADL = cumulative sum of (CLV * Volume)

        Expects: ['High','Low','Close','Volume']
        """
        if not all(col in data.columns
                   for col in ['High', 'Low', 'Close', 'Volume']):
            raise ValueError(
                "Data must contain High, Low, Close, Volume for ADL.")

        denom = (data['High'] - data['Low']).replace(0, np.nan)
        clv = ((data['Close'] - data['Low']) -
               (data['High'] - data['Close'])) / denom
        clv = clv.fillna(0)
        adl = (clv * data['Volume']).cumsum()

        return pd.Series(adl, index=data.index, name='ADL')

    @staticmethod
    def add_lagged_features(data: pd.DataFrame,
                            columns: List[str],
                            lags: int = 3) -> pd.DataFrame:
        """
        Adds lagged versions of given columns. e.g. Close_lag1, Close_lag2, ...
        """
        for col in columns:
            if col not in data.columns:
                logger.warning(
                    f"Column '{col}' not found for lagged features.")
                continue
            for lag in range(1, lags + 1):
                data[f"{col}_lag{lag}"] = data[col].shift(lag)
        return data

    @staticmethod
    def add_rolling_features(data: pd.DataFrame, column: str,
                             windows: List[int]) -> pd.DataFrame:
        """
        Adds rolling mean, std, min, max for each window in 'windows'.
        """
        if column not in data.columns:
            logger.warning(
                f"Column '{column}' not found for rolling features.")
            return data

        for w in windows:
            data[f"{column}_rolling_mean_{w}"] = data[column].rolling(
                window=w).mean()
            data[f"{column}_rolling_std_{w}"] = data[column].rolling(
                window=w).std()
            data[f"{column}_rolling_min_{w}"] = data[column].rolling(
                window=w).min()
            data[f"{column}_rolling_max_{w}"] = data[column].rolling(
                window=w).max()
        return data

    @staticmethod
    def add_fourier_transform(prices: pd.Series,
                              top_n: int = 3,
                              window: int = 30) -> pd.DataFrame:
        """
        Adds top_n Fourier Transform coefficients using rolling windows.
        """
        if len(prices) < window:
            logger.warning(
                f"Data length {len(prices)} < window size {window}. Returning empty."
            )
            return pd.DataFrame(index=prices.index)

        result = pd.DataFrame(index=prices.index)

        try:
            for i in range(window, len(prices) + 1):
                window_data = prices.iloc[i - window:i]
                try:
                    fft_result = fft(window_data.values)
                    for j in range(top_n):
                        result.loc[prices.index[i - 1], f'FFT_{j + 1}'] = np.abs(fft_result[j])
                except Exception as e:
                    logger.error(f"FFT calculation error in window: {e}")

            return result.fillna(method='ffill')
        except Exception as e:
            logger.error(f"FFT calculation error: {e}")
            return pd.DataFrame(index=prices.index)

    def _calculate_symbol_correlation(self,
                                      ref: pd.Series,
                                      other: pd.Series,
                                      min_points: int = 2) -> Optional[float]:
        aligned = pd.concat([ref, other], axis=1, join='inner').dropna()
        if len(aligned) >= min_points:
            return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        else:
            return None

    def _calculate_all_correlations(
            self, symbol: str, ref_data: pd.Series,
            portfolio_data: Dict[str,
                                 pd.DataFrame]) -> Dict[str, Optional[float]]:
        """
        Computes correlation of 'ref_data' with each symbol's 'Close' in portfolio_data.
        Parallel if self.n_jobs > 1.
        """
        tasks = []
        for other_symbol, other_df in portfolio_data.items():
            if other_symbol == symbol:
                continue
            if 'Close' not in other_df.columns:
                logger.warning(
                    f"[{other_symbol}] No 'Close' column for correlation.")
                continue
            tasks.append((other_symbol, other_df['Close']))

        correlations = {}
        if self.n_jobs > 1:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.n_jobs) as executor:
                future_map = {
                    executor.submit(self._calculate_symbol_correlation, ref_data, other_close):
                    other_sym
                    for (other_sym, other_close) in tasks
                }
                for future in concurrent.futures.as_completed(future_map):
                    sym = future_map[future]
                    correlations[sym] = future.result()
        else:
            # Single-thread fallback
            for other_sym, other_close in tasks:
                correlations[other_sym] = self._calculate_symbol_correlation(
                    ref_data, other_close)

        return correlations

    def prepare_data(
            self,
            portfolio_data: Dict[str,
                                 pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Main pipeline to:
         - Validate data
         - Calculate standard indicators
         - Add FFT
         - Correlate with other symbols
         - Normalize non-OHLCV columns
         - Drop rows with missing data

        Returns a dict of prepared DataFrames keyed by symbol.
        """
        if not portfolio_data:
            raise ValueError(
                "No data available. Please provide 'portfolio_data' first.")

        if not isinstance(portfolio_data, dict):
            raise TypeError(
                "'portfolio_data' must be a dictionary of DataFrames.")

        prepared_data = {}
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

        for symbol, df in portfolio_data.items():
            try:
                # Basic check: must have a 'Close' column
                if not validate_dataframe(df, ['Close']):
                    logger.error(
                        f"[{symbol}] Invalid DataFrame format. Missing 'Close' or empty."
                    )
                    continue
                # Make a copy so we don't mutate the original
                data_copy = df.copy()

                # Must have at least 50 data points for meaningful rolling calculations
                if len(data_copy) < 50:
                    logger.warning(
                        f"[{symbol}] Insufficient data (<50 rows). Skipping advanced features."
                    )
                    prepared_data[symbol] = data_copy
                    continue

                # -----------------------------------------------------------------
                # 1) Compute Indicators
                # -----------------------------------------------------------------
                try:
                    data_copy['RSI'] = self.calculate_rsi(data_copy['Close'])
                    data_copy['Volatility'] = self.calculate_volatility(
                        data_copy['Close'])

                    bollinger_bands = self.calculate_bollinger_bands(
                        data_copy['Close'])
                    data_copy = pd.concat([data_copy, bollinger_bands], axis=1)

                    macd_df = self.calculate_macd(data_copy['Close'])
                    data_copy = pd.concat([data_copy, macd_df], axis=1)

                    # Some advanced indicators require columns: 'High','Low','Volume'
                    if all(col in data_copy.columns
                           for col in ['High', 'Low', 'Close']):
                        stoch = self.calculate_stochastic_oscillator(
                            data_copy[['High', 'Low', 'Close']])
                        data_copy = pd.concat([data_copy, stoch], axis=1)

                    if all(col in data_copy.columns
                           for col in ['High', 'Low', 'Close', 'Volume']):
                        mfi = self.calculate_money_flow_index(
                            data_copy[['High', 'Low', 'Close', 'Volume']])
                        data_copy['MFI'] = mfi

                        obv = self.calculate_on_balance_volume(
                            data_copy[['Close', 'Volume']])
                        data_copy['OBV'] = obv

                        adl = self.calculate_accumulation_distribution_line(
                            data_copy[['High', 'Low', 'Close', 'Volume']])
                        data_copy['ADL'] = adl
                except Exception as e:
                    logger.error(
                        f"[{symbol}] Error in indicator calculations: {e}")

                # -----------------------------------------------------------------
                # 2) Lagged & Rolling Features
                # -----------------------------------------------------------------
                try:
                    data_copy = self.add_lagged_features(data_copy,
                                                         ['Close', 'RSI'],
                                                         lags=5)
                    data_copy = self.add_rolling_features(data_copy,
                                                          'Close',
                                                          windows=[7, 14, 30])
                except Exception as e:
                    logger.error(
                        f"[{symbol}] Error in lagged/rolling features: {e}")

                # -----------------------------------------------------------------
                # 3) Fourier Transform
                # -----------------------------------------------------------------
                try:
                    fft_df = self.add_fourier_transform(data_copy['Close'],
                                                        top_n=3)
                    if not fft_df.empty:
                        data_copy = pd.concat([data_copy, fft_df], axis=1)
                    else:
                        logger.warning(
                            f"[{symbol}] Skipping FFT. Insufficient data or error."
                        )
                except Exception as e:
                    logger.error(f"[{symbol}] FFT Error: {e}")

                # -----------------------------------------------------------------
                # 4) Correlations
                # -----------------------------------------------------------------
                try:
                    correlations = self._calculate_all_correlations(
                        symbol, data_copy['Close'], portfolio_data)
                    data_copy['Correlations'] = str(correlations)
                except Exception as e:
                    logger.error(
                        f"[{symbol}] Error calculating correlations: {e}")

                # -----------------------------------------------------------------
                # 5) Normalization (non-OHLCV columns & non-object)
                # -----------------------------------------------------------------
                ohlcv_backup = data_copy[ohlcv_cols].copy() if all(
                    c in data_copy.columns for c in ohlcv_cols) else None

                for col in data_copy.columns:
                    if col in ohlcv_cols:
                        continue
                    if data_copy[col].dtype == 'object':
                        continue
                    # only normalize if >1 unique value
                    if data_copy[col].nunique() > 1:
                        data_copy[col] = self.normalize_data(data_copy[col])
                    else:
                        logger.warning(
                            f"[{symbol}] Column '{col}' is constant or insufficiently variable for normalization."
                        )

                # Restore the original OHLCV data if available
                if ohlcv_backup is not None:
                    for c in ohlcv_cols:
                        data_copy[c] = ohlcv_backup[c]

                # -----------------------------------------------------------------
                # 6) Drop NaNs
                # -----------------------------------------------------------------
                data_copy.dropna(inplace=True)
                if data_copy.empty:
                    logger.warning(
                        f"[{symbol}] All rows dropped after dropping NaNs.")

                prepared_data[symbol] = data_copy

            except Exception as ex:
                logger.error(
                    f"[{symbol}] Unexpected error in prepare_data: {ex}")
                prepared_data[symbol] = df  # fallback to the original if error

        return prepared_data