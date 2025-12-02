"""
Abstract base class for prediction sources.

All prediction models must inherit from BasePredictionSource and implement
the required methods for training and prediction.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class PredictionHorizon(Enum):
    """Standard prediction horizons."""
    DAY_1 = 1      # Next day
    DAY_5 = 5      # 1 week
    DAY_21 = 21    # 1 month
    DAY_63 = 63    # 1 quarter


class PredictionMode(Enum):
    """Types of predictions a source can provide."""
    PRICE = "price"              # Price magnitude prediction
    DIRECTION = "direction"      # Up/down direction (1/-1)
    VOLATILITY = "volatility"    # Expected volatility
    CONFIDENCE = "confidence"    # Prediction confidence [0-1]


@dataclass
class PredictionOutput:
    """Container for prediction outputs."""
    values: np.ndarray           # Prediction values
    horizon: int                 # Prediction horizon in days
    mode: PredictionMode         # Type of prediction
    symbol: str                  # Symbol this prediction is for
    timestamps: Optional[pd.DatetimeIndex] = None


class BasePredictionSource(ABC):
    """Abstract base class for all prediction sources.

    Critical constraint: At market open on day T, only the following data is available:
    - Open price of day T (current day)
    - High, Low, Close, Volume of day T-1 (previous day)

    All prediction sources must respect this point-in-time correctness.
    """

    # Available columns at prediction time (Open is current day, HLCV is previous day)
    AVAILABLE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

    def __init__(self, name: Optional[str] = None):
        """Initialize prediction source.

        Args:
            name: Source name (defaults to class name)
        """
        self._name = name or self.__class__.__name__
        self._is_trained = False
        self._training_params: Dict[str, Any] = {}
        self._horizons: List[int] = [1, 5, 21]
        self._modes: List[PredictionMode] = [
            PredictionMode.PRICE,
            PredictionMode.DIRECTION,
            PredictionMode.VOLATILITY,
            PredictionMode.CONFIDENCE,
        ]

    @property
    def name(self) -> str:
        """Get source name."""
        return self._name

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained

    @property
    def horizons(self) -> List[int]:
        """Get prediction horizons."""
        return self._horizons

    @horizons.setter
    def horizons(self, value: List[int]) -> None:
        """Set prediction horizons."""
        self._horizons = sorted(value)

    @property
    def modes(self) -> List[PredictionMode]:
        """Get prediction modes."""
        return self._modes

    @abstractmethod
    def train(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        start_idx: int,
        end_idx: int,
    ) -> None:
        """Train the model on a window of data.

        Args:
            data: Full DataFrame with OHLCV data
            symbols: List of symbols to train on
            start_idx: Start index of training window
            end_idx: End index of training window (exclusive)
        """
        pass

    @abstractmethod
    def predict(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        current_idx: int,
    ) -> Dict[str, Dict[int, Dict[PredictionMode, float]]]:
        """Generate predictions for current time step.

        Args:
            data: Full DataFrame with OHLCV data
            symbols: List of symbols to predict
            current_idx: Current time index for point-in-time prediction

        Returns:
            Nested dict: {symbol: {horizon: {mode: value}}}
        """
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Directory path to save model
        """
        pass

    @abstractmethod
    def load_model(self, path: str) -> bool:
        """Load model from disk.

        Args:
            path: Directory path to load model from

        Returns:
            True if model loaded successfully
        """
        pass

    def get_feature_columns(self, symbols: List[str]) -> List[str]:
        """Get list of feature column names this source produces.

        Args:
            symbols: List of symbols

        Returns:
            List of column names in format: {source}_{symbol}_{horizon}d_{mode}
        """
        columns = []
        for symbol in symbols:
            for horizon in self.horizons:
                for mode in self.modes:
                    col_name = f"{self.name}_{symbol}_{horizon}d_{mode.value}"
                    columns.append(col_name)
        return columns

    def _prepare_features(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        end_idx: int,
        sequence_length: int = 20,
    ) -> Optional[np.ndarray]:
        """Prepare feature array for model input.

        Respects point-in-time correctness:
        - At time T, we use Open_T and HLCV from T-1

        Args:
            data: Full DataFrame
            symbols: Symbols to include
            end_idx: End index (current time step)
            sequence_length: Number of time steps to include

        Returns:
            Feature array of shape (sequence_length, num_features) or None if insufficient data
        """
        if end_idx < sequence_length:
            return None

        start_idx = end_idx - sequence_length
        features = []

        for t in range(start_idx, end_idx):
            row_features = []
            for symbol in symbols:
                # Open from current day
                open_col = f'Open_{symbol}'
                if open_col in data.columns:
                    row_features.append(data.iloc[t][open_col])

                # HLCV from previous day (shifted by 1 in the data)
                for col_type in ['High', 'Low', 'Close', 'Volume']:
                    col_name = f'{col_type}_{symbol}'
                    if col_name in data.columns:
                        row_features.append(data.iloc[t][col_name])

            features.append(row_features)

        feature_array = np.array(features, dtype=np.float32)

        # Handle NaN values
        feature_array = np.nan_to_num(feature_array, nan=0.0)

        return feature_array

    def _compute_targets(
        self,
        data: pd.DataFrame,
        symbol: str,
        current_idx: int,
        horizon: int,
    ) -> Optional[Dict[PredictionMode, float]]:
        """Compute target values for training.

        Args:
            data: Full DataFrame
            symbol: Symbol to compute targets for
            current_idx: Current time index
            horizon: Prediction horizon in days

        Returns:
            Dict of targets by mode, or None if insufficient data
        """
        target_idx = current_idx + horizon
        close_col = f'Close_{symbol}'

        if target_idx >= len(data) or close_col not in data.columns:
            return None

        current_close = data.iloc[current_idx][close_col]
        future_close = data.iloc[target_idx][close_col]

        if pd.isna(current_close) or pd.isna(future_close) or current_close == 0:
            return None

        # Price change (return)
        price_change = (future_close - current_close) / current_close

        # Direction (1 for up, -1 for down)
        direction = 1.0 if future_close > current_close else -1.0

        # Volatility (std of returns in horizon window)
        if target_idx + 1 <= len(data):
            returns = data[close_col].iloc[current_idx:target_idx+1].pct_change().dropna()
            volatility = float(returns.std()) if len(returns) > 0 else 0.0
        else:
            volatility = 0.0

        return {
            PredictionMode.PRICE: price_change,
            PredictionMode.DIRECTION: direction,
            PredictionMode.VOLATILITY: volatility,
            PredictionMode.CONFIDENCE: 1.0,  # Target confidence is always 1 during training
        }
