"""
Feature engineering pipeline with consolidated technical indicators.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List
from scipy.fftpack import fft

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class FeatureEngineer:
    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs

    @staticmethod
    def normalize_data(data: pd.Series) -> pd.Series:
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

    def prepare_data(self, portfolio_data: pd.DataFrame) -> pd.DataFrame:
        """
        Main pipeline to prepare features from concatenated portfolio data.
        Columns are expected to be named like: Open_AAPL, Close_AAPL, etc.
        """
        if not isinstance(portfolio_data, pd.DataFrame):
            raise TypeError("portfolio_data must be a DataFrame")
        if portfolio_data.empty:
            raise ValueError("portfolio_data is empty")

        # Get unique symbols from column names
        symbols = set(col.split('_')[1] for col in portfolio_data.columns if '_' in col)

        prepared_data = portfolio_data.copy()

        for symbol in symbols:
            try:
                # Calculate indicators for each symbol using its specific columns
                close_col = f'Close_{symbol}'
                high_col = f'High_{symbol}'
                low_col = f'Low_{symbol}'
                volume_col = f'Volume_{symbol}'

                if close_col in portfolio_data.columns:
                    prepared_data[f'RSI_{symbol}'] = self.calculate_rsi(portfolio_data[close_col])
                    prepared_data[f'Volatility_{symbol}'] = self.calculate_volatility(portfolio_data[close_col])

                    # Bollinger Bands
                    bb = self.calculate_bollinger_bands(portfolio_data[close_col])
                    prepared_data[f'BB_Upper_{symbol}'] = bb['Upper_Band']
                    prepared_data[f'BB_Lower_{symbol}'] = bb['Lower_Band']
                    prepared_data[f'BB_SMA_{symbol}'] = bb['SMA']

                    # MACD
                    macd = self.calculate_macd(portfolio_data[close_col])
                    prepared_data[f'MACD_{symbol}'] = macd['MACD']
                    prepared_data[f'MACD_Signal_{symbol}'] = macd['Signal']

                    # Add lagged features
                    for lag in range(1, 6):
                        prepared_data[f'{close_col}_lag{lag}'] = portfolio_data[close_col].shift(lag)

                    # Add rolling features
                    for window in [7, 14, 30]:
                        prepared_data[f'{close_col}_rolling_mean_{window}'] = portfolio_data[close_col].rolling(window=window).mean()
                        prepared_data[f'{close_col}_rolling_std_{window}'] = portfolio_data[close_col].rolling(window=window).std()

                    # FFT features
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

        # Drop any NaN values that might have been introduced
        prepared_data.dropna(inplace=True)

        return prepared_data