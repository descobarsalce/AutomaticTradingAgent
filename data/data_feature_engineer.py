
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from scipy.fftpack import fft

logger = logging.getLogger(__name__)

def validate_numeric(value: float, min_value: Optional[float] = None, max_value: Optional[float] = None) -> bool:
    return isinstance(value, (int, float)) and (min_value is None or value >= min_value) and (max_value is None or value <= max_value)

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty and all(col in df.columns for col in required_columns)

def validate_portfolio_weights(weights: Dict[str, float]) -> bool:
    return isinstance(weights, dict) and all(isinstance(w, (int, float)) and 0 <= w <= 1 for w in weights.values()) and abs(sum(weights.values()) - 1.0) < 1e-6

def normalize_data(data: pd.Series) -> pd.Series:
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val) if max_val > min_val else pd.Series(0, index=data.index)

class FeatureEngineer:
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20) -> pd.DataFrame:
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        return pd.DataFrame({
            'Upper_Band': sma + (2 * std),
            'Lower_Band': sma - (2 * std),
            'SMA': sma
        })

    @staticmethod
    def calculate_macd(prices: pd.Series, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
        ema_short = prices.ewm(span=short_window, adjust=False).mean()
        ema_long = prices.ewm(span=long_window, adjust=False).mean()
        macd = ema_short - ema_long
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return pd.DataFrame({'MACD': macd, 'Signal': signal})

    @staticmethod
    def add_lagged_features(data: pd.DataFrame, columns: list, lags: int = 3) -> pd.DataFrame:
        for col in columns:
            for lag in range(1, lags + 1):
                data[f"{col}_lag{lag}"] = data[col].shift(lag)
        return data

    @staticmethod
    def add_rolling_features(data: pd.DataFrame, column: str, windows: list) -> pd.DataFrame:
        for window in windows:
            data[f"{column}_rolling_mean_{window}"] = data[column].rolling(window=window).mean()
            data[f"{column}_rolling_std_{window}"] = data[column].rolling(window=window).std()
            data[f"{column}_rolling_min_{window}"] = data[column].rolling(window=window).min()
            data[f"{column}_rolling_max_{window}"] = data[column].rolling(window=window).max()
        return data

    @staticmethod
    def calculate_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
        return prices.pct_change().rolling(window=window).std()

    @staticmethod
    def add_fourier_transform(data: pd.Series, top_n: int = 5) -> pd.DataFrame:
        fft_result = fft(data.values)
        fft_df = pd.DataFrame({
            f'FFT_{i+1}': np.abs(fft_result[i]) for i in range(top_n)
        }, index=[0])
        return pd.concat([data.reset_index(drop=True), fft_df], axis=1)

    def prepare_data(self, portfolio_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if not portfolio_data:
            raise ValueError("No data available. Please fetch data first.")
        
        if not isinstance(portfolio_data, dict):
            raise TypeError("Portfolio data must be a dictionary of DataFrames")
            
        prepared_data = {}
        for symbol, data in portfolio_data.items():
            try:
                if not validate_dataframe(data, ['Close']):
                    logger.error(f"Invalid data format for {symbol}")
                    continue
                
                prepared_df = data.copy()
                
                if len(prepared_df) >= 50:
                    try:
                        # Calculate technical indicators
                        prepared_df['RSI'] = self.calculate_rsi(prepared_df['Close'])
                        prepared_df['Volatility'] = self.calculate_volatility(prepared_df['Close'])
                        
                        bollinger_bands = self.calculate_bollinger_bands(prepared_df['Close'])
                        prepared_df = pd.concat([prepared_df, bollinger_bands], axis=1)
                        
                        macd = self.calculate_macd(prepared_df['Close'])
                        prepared_df = pd.concat([prepared_df, macd], axis=1)

                        # Add lagged and rolling features
                        prepared_df = self.add_lagged_features(prepared_df, ['Close', 'RSI'], lags=5)
                        prepared_df = self.add_rolling_features(prepared_df, 'Close', windows=[7, 14, 30])

                        # Store original OHLCV data
                        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                        ohlcv_data = prepared_df[ohlcv_cols].copy()
                        
                        # Add Fourier Transform features
                        fft_df = self.add_fourier_transform(prepared_df['Close'])
                        prepared_df = pd.concat([prepared_df, fft_df], axis=1)
                        
                        # Calculate correlations with other symbols
                        correlations = {}
                        for other_symbol, other_data in portfolio_data.items():
                            if other_symbol != symbol and validate_dataframe(other_data, ['Close']):
                                correlations[other_symbol] = prepared_df['Close'].corr(other_data['Close'])
                        prepared_df['Correlations'] = str(correlations)
                        
                        # Normalize features except OHLCV
                        for col in prepared_df.columns:
                            if col not in ohlcv_cols and prepared_df[col].dtype != 'object':
                                prepared_df[col] = normalize_data(prepared_df[col])
                                
                        # Restore original OHLCV data
                        for col in ohlcv_cols:
                            prepared_df[col] = ohlcv_data[col]
                        
                        # Handle missing values
                        prepared_df = prepared_df.dropna()
                        
                        if not prepared_df.empty:
                            prepared_data[symbol] = prepared_df
                        else:
                            logger.warning(f"No valid data points remaining for {symbol} after calculations")
                    except Exception as e:
                        logger.error(f"Error calculating indicators for {symbol}: {str(e)}")
                        prepared_data[symbol] = data
                else:
                    logger.warning(f"Insufficient data points for {symbol} (minimum 50 required)")
                    prepared_data[symbol] = data
                    
            except Exception as e:
                logger.error(f"Error preparing data for {symbol}: {str(e)}")
                prepared_data[symbol] = data
                
        return prepared_data
