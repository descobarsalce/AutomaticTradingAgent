"""
Prediction engine for orchestrating ML predictions.

Handles training, prediction generation, and integration with feature pipeline.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from data.feature_engineering.predictions.base_prediction import (
    BasePredictionSource,
    PredictionMode,
)
from data.feature_engineering.predictions.prediction_registry import PredictionRegistry

logger = logging.getLogger(__name__)


class PredictionEngine:
    """Orchestrates prediction computation and training.

    Responsibilities:
    - Managing training across all prediction sources
    - Generating predictions for the trading environment
    - Rolling window (walk-forward) training
    """

    def __init__(
        self,
        prediction_registry: Optional[PredictionRegistry] = None,
        train_window_days: int = 252,
        retrain_frequency_days: int = 21,
    ):
        """Initialize prediction engine.

        Args:
            prediction_registry: Registry containing prediction sources
            train_window_days: Number of days for training window
            retrain_frequency_days: How often to retrain (in days)
        """
        self._registry = prediction_registry or PredictionRegistry()
        self._train_window_days = train_window_days
        self._retrain_frequency_days = retrain_frequency_days
        self._last_train_idx: Dict[str, int] = {}

    @property
    def registry(self) -> PredictionRegistry:
        """Get prediction registry."""
        return self._registry

    @property
    def train_window_days(self) -> int:
        """Get training window size."""
        return self._train_window_days

    @train_window_days.setter
    def train_window_days(self, value: int) -> None:
        """Set training window size."""
        self._train_window_days = max(30, value)

    def train_source(
        self,
        source_name: str,
        data: pd.DataFrame,
        symbols: List[str],
        end_idx: Optional[int] = None,
    ) -> bool:
        """Train a specific prediction source.

        Args:
            source_name: Name of source to train
            data: Full DataFrame with OHLCV data
            symbols: List of symbols
            end_idx: End index for training (defaults to all available data)

        Returns:
            True if training succeeded
        """
        source = self._registry.get_source(source_name)
        if source is None:
            logger.error(f"Source not found: {source_name}")
            return False

        if end_idx is None:
            end_idx = len(data)

        start_idx = max(0, end_idx - self._train_window_days)

        if end_idx - start_idx < 30:
            logger.warning(f"Insufficient data for training {source_name}")
            return False

        try:
            logger.info(
                f"Training {source_name} on indices {start_idx} to {end_idx}"
            )
            source.train(data, symbols, start_idx, end_idx)
            self._last_train_idx[source_name] = end_idx

            # Save model after training
            self._registry.save_source_model(source_name)

            return True
        except Exception as e:
            logger.error(f"Training failed for {source_name}: {e}")
            return False

    def train_all_untrained(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        end_idx: Optional[int] = None,
    ) -> int:
        """Train all untrained sources.

        Args:
            data: Full DataFrame with OHLCV data
            symbols: List of symbols
            end_idx: End index for training

        Returns:
            Number of sources successfully trained
        """
        trained = 0
        for source in self._registry.get_untrained_sources():
            if self.train_source(source.name, data, symbols, end_idx):
                trained += 1
        return trained

    def train_rolling(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        current_idx: int,
        force: bool = False,
    ) -> int:
        """Perform rolling window training if needed.

        Args:
            data: Full DataFrame with OHLCV data
            symbols: List of symbols
            current_idx: Current time index
            force: Force retraining even if not due

        Returns:
            Number of sources retrained
        """
        retrained = 0

        for source in self._registry.get_all_sources():
            last_train = self._last_train_idx.get(source.name, -1)

            # Check if retraining is needed
            needs_retrain = (
                force
                or not source.is_trained
                or (current_idx - last_train) >= self._retrain_frequency_days
            )

            if needs_retrain:
                if self.train_source(source.name, data, symbols, current_idx):
                    retrained += 1

        return retrained

    def compute_predictions(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        current_idx: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute predictions from all trained sources.

        Args:
            data: Full DataFrame with OHLCV data
            symbols: List of symbols
            current_idx: Current time index (defaults to last valid index)

        Returns:
            DataFrame with prediction columns added
        """
        if current_idx is None:
            current_idx = len(data) - 1

        result = data.copy()

        # Initialize prediction columns with NaN
        all_columns = self._registry.get_all_feature_columns(symbols)
        for col in all_columns:
            if col not in result.columns:
                result[col] = np.nan

        # Get predictions from each trained source
        for source in self._registry.get_trained_sources():
            try:
                predictions = source.predict(data, symbols, current_idx)

                # Fill in prediction values
                for symbol, horizon_dict in predictions.items():
                    for horizon, mode_dict in horizon_dict.items():
                        for mode, value in mode_dict.items():
                            col_name = f"{source.name}_{symbol}_{horizon}d_{mode.value}"
                            if col_name in result.columns:
                                result.loc[result.index[current_idx], col_name] = value

            except Exception as e:
                logger.error(f"Prediction failed for {source.name}: {e}")

        return result

    def compute_predictions_batch(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        start_idx: int,
        end_idx: int,
    ) -> pd.DataFrame:
        """Compute predictions for a range of indices.

        Args:
            data: Full DataFrame with OHLCV data
            symbols: List of symbols
            start_idx: Start index
            end_idx: End index (exclusive)

        Returns:
            DataFrame with prediction columns for all indices
        """
        result = data.copy()

        # Initialize prediction columns
        all_columns = self._registry.get_all_feature_columns(symbols)
        for col in all_columns:
            if col not in result.columns:
                result[col] = np.nan

        # Compute predictions for each index
        for idx in range(start_idx, end_idx):
            for source in self._registry.get_trained_sources():
                try:
                    predictions = source.predict(data, symbols, idx)

                    for symbol, horizon_dict in predictions.items():
                        for horizon, mode_dict in horizon_dict.items():
                            for mode, value in mode_dict.items():
                                col_name = f"{source.name}_{symbol}_{horizon}d_{mode.value}"
                                if col_name in result.columns:
                                    result.loc[result.index[idx], col_name] = value

                except Exception as e:
                    logger.error(f"Batch prediction failed at idx {idx}: {e}")

        return result

    def get_prediction_columns(self, symbols: List[str]) -> List[str]:
        """Get list of all prediction column names.

        Args:
            symbols: List of symbols

        Returns:
            List of column names
        """
        return self._registry.get_all_feature_columns(symbols)

    def get_source_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered sources.

        Returns:
            Dict with source info
        """
        info = {}
        for source in self._registry.get_all_sources():
            info[source.name] = {
                'is_trained': source.is_trained,
                'horizons': source.horizons,
                'modes': [m.value for m in source.modes],
                'last_train_idx': self._last_train_idx.get(source.name),
            }
        return info
