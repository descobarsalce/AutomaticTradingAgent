
"""
Base technical indicators implementation to be shared across the platform.
Eliminates duplicate calculations between feature engineering and metrics.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class BaseTechnicalIndicators:
    """Base class for technical indicators used across the platform."""
    
    @staticmethod
    def calculate_rsi(data: Union[pd.Series, np.ndarray], period: int = 14) -> Union[pd.Series, np.ndarray]:
        """Relative Strength Index (RSI) calculation"""
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        loss = loss.replace(0, 1e-10)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        result = rsi.clip(lower=0, upper=100)
        
        return result if isinstance(data, pd.Series) else result.to_numpy()

    @staticmethod
    def calculate_bollinger_bands(
        data: Union[pd.Series, np.ndarray],
        window: int = 20,
        num_std: float = 2.0
    ) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
        """Bollinger Bands calculation"""
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
            
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        if isinstance(data, pd.Series):
            return sma, upper_band, lower_band
        return sma.to_numpy(), upper_band.to_numpy(), lower_band.to_numpy()

    @staticmethod
    def calculate_macd(
        data: Union[pd.Series, np.ndarray],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
        """MACD calculation"""
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
            
        fast_ema = data.ewm(span=fast_period, adjust=False).mean()
        slow_ema = data.ewm(span=slow_period, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        
        if isinstance(data, pd.Series):
            return macd, signal
        return macd.to_numpy(), signal.to_numpy()

    @staticmethod
    def calculate_volatility(data: Union[pd.Series, np.ndarray], window: int = 20) -> Union[pd.Series, np.ndarray]:
        """Volatility calculation"""
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
            
        volatility = data.pct_change().rolling(window=window).std()
        return volatility if isinstance(data, pd.Series) else volatility.to_numpy()
