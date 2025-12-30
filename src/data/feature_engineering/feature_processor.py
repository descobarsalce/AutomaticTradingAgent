"""
Feature processor for integrating feature engineering with trading environment.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """Processes raw market data into features for the trading model.

    This class integrates the modular feature engineering system with
    the trading environment, handling feature computation and normalization.
    """

    def __init__(
        self,
        feature_config: Optional[Dict[str, Any]] = None,
        symbols: Optional[List[str]] = None,
    ):
        """Initialize feature processor.

        Args:
            feature_config: Feature configuration from the UI
            symbols: List of stock symbols
        """
        self.feature_config = feature_config or {}
        self.symbols = symbols or []
        self.engineer = None
        self.feature_columns: List[str] = []
        self._normalization_params: Dict[str, Dict[str, float]] = {}
        self._initialized = False

        # Prediction layer
        self.prediction_engine = None
        self.prediction_columns: List[str] = []
        self._prediction_normalization_params: Dict[str, Dict[str, float]] = {}

    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize feature processor with data.

        Args:
            data: Historical market data for fitting normalization
        """
        if not self.feature_config.get('use_feature_engineering', False):
            self._initialized = True
            return

        try:
            from src.data.feature_engineering import FeatureEngineer

            self.engineer = FeatureEngineer(enable_cache=True)
            self.engineer.register_default_sources()

            # Compute features on historical data to fit normalization
            selected_features = self._get_selected_feature_list()
            if selected_features:
                features_df = self.engineer.compute_features(
                    data,
                    symbols=self.symbols,
                    features=selected_features,
                    drop_na=False,
                )

                # Store feature columns
                self.feature_columns = [
                    col for col in features_df.columns
                    if col not in data.columns
                ]

                # Compute normalization parameters
                if self.feature_config.get('normalize_features', True):
                    self._compute_normalization_params(features_df)

            self._initialized = True
            logger.info(
                f"Feature processor initialized with {len(self.feature_columns)} features"
            )

            # Initialize prediction layer if configured
            if self.feature_config.get('use_predictions', False):
                self._initialize_predictions(data)

        except Exception as e:
            logger.error(f"Failed to initialize feature processor: {e}")
            self._initialized = False
            raise

    def _initialize_predictions(self, data: pd.DataFrame) -> None:
        """Initialize prediction engine and train models.

        Args:
            data: Historical market data for training
        """
        try:
            from src.data.feature_engineering.predictions import (
                PredictionEngine,
                PredictionRegistry,
            )
            from src.data.feature_engineering.predictions.sources import LSTMPredictor

            pred_config = self.feature_config.get('predictions', {})
            models_dir = pred_config.get('models_dir', 'artifacts/models')
            train_window = pred_config.get('train_window_days', 252)
            horizons = pred_config.get('horizons', [1, 5, 21])

            # Create registry and engine
            registry = PredictionRegistry(models_dir=models_dir)
            self.prediction_engine = PredictionEngine(
                prediction_registry=registry,
                train_window_days=train_window,
            )

            # Register configured sources
            sources_config = pred_config.get('sources', {})
            if sources_config.get('LSTMPredictor', {}).get('enabled', False):
                lstm_config = sources_config['LSTMPredictor']
                predictor = LSTMPredictor(
                    sequence_length=lstm_config.get('sequence_length', 20),
                    lstm_units=lstm_config.get('lstm_units', 64),
                    horizons=horizons,
                )
                registry.register_source(predictor)

            # Train untrained models
            self.prediction_engine.train_all_untrained(data, self.symbols)

            # Store prediction columns
            self.prediction_columns = self.prediction_engine.get_prediction_columns(
                self.symbols
            )

            # Compute normalization params for predictions
            if self.feature_config.get('normalize_predictions', True):
                pred_data = self.prediction_engine.compute_predictions_batch(
                    data, self.symbols,
                    start_idx=max(0, len(data) - train_window),
                    end_idx=len(data),
                )
                self._compute_prediction_normalization_params(pred_data)

            logger.info(
                f"Prediction layer initialized with {len(self.prediction_columns)} columns"
            )

        except ImportError as e:
            logger.warning(f"Prediction layer not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize prediction layer: {e}")

    def _compute_prediction_normalization_params(self, data: pd.DataFrame) -> None:
        """Compute normalization parameters for prediction columns."""
        for col in self.prediction_columns:
            if col in data.columns:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    self._prediction_normalization_params[col] = {
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()) or 1.0,
                    }

    def _get_selected_feature_list(self) -> List[str]:
        """Get flat list of selected feature names."""
        features = []
        sources = self.feature_config.get('sources', {})
        for source_name, src_config in sources.items():
            if src_config.get('enabled', False):
                features.extend(src_config.get('features', []))
        return features

    def _compute_normalization_params(self, data: pd.DataFrame) -> None:
        """Compute normalization parameters from src.data."""
        for col in self.feature_columns:
            if col in data.columns:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    self._normalization_params[col] = {
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()) or 1.0,
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                    }

    def compute_features(
        self,
        data: pd.DataFrame,
        current_index: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute features for given data.

        Args:
            data: Market data DataFrame
            current_index: If provided, only compute up to this index

        Returns:
            DataFrame with computed features
        """
        if not self._initialized:
            raise RuntimeError("Feature processor not initialized")

        if not self.feature_config.get('use_feature_engineering', False):
            return data

        if self.engineer is None:
            return data

        try:
            selected_features = self._get_selected_feature_list()
            if not selected_features:
                return data

            features_df = self.engineer.compute_features(
                data,
                symbols=self.symbols,
                features=selected_features,
                current_index=current_index,
                drop_na=False,
            )

            # Normalize if configured
            if self.feature_config.get('normalize_features', True):
                features_df = self._normalize_features(features_df)

            # Add predictions if configured
            if (self.feature_config.get('use_predictions', False)
                    and self.prediction_engine is not None):
                features_df = self._compute_predictions(features_df, current_index)

            return features_df

        except Exception as e:
            logger.error(f"Error computing features: {e}")
            return data

    def _compute_predictions(
        self,
        data: pd.DataFrame,
        current_index: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute predictions and add to DataFrame.

        Args:
            data: DataFrame with features
            current_index: Current time index

        Returns:
            DataFrame with prediction columns added
        """
        if self.prediction_engine is None:
            return data

        try:
            result = self.prediction_engine.compute_predictions(
                data, self.symbols, current_index
            )

            # Normalize predictions if configured
            if self.feature_config.get('normalize_predictions', True):
                result = self._normalize_predictions(result)

            return result
        except Exception as e:
            logger.error(f"Error computing predictions: {e}")
            return data

    def _normalize_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize prediction columns."""
        result = data.copy()

        for col in self.prediction_columns:
            if col in result.columns and col in self._prediction_normalization_params:
                params = self._prediction_normalization_params[col]
                result[col] = (result[col] - params['mean']) / params['std']
                result[col] = result[col].clip(-10, 10)

        return result

    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize feature columns."""
        result = data.copy()

        for col in self.feature_columns:
            if col in result.columns and col in self._normalization_params:
                params = self._normalization_params[col]
                # Z-score normalization
                result[col] = (result[col] - params['mean']) / params['std']
                # Clip to reasonable range
                result[col] = result[col].clip(-10, 10)

        return result

    def get_observation_vector(
        self,
        data: pd.DataFrame,
        current_index: int,
        positions: Dict[str, float],
        balance: float,
    ) -> np.ndarray:
        """Get observation vector for the current step.

        Args:
            data: Full market data DataFrame
            current_index: Current time step index
            positions: Current portfolio positions
            balance: Current cash balance

        Returns:
            Flattened observation vector
        """
        obs_components = []

        # Add feature values for current step
        if self.feature_config.get('use_feature_engineering', False):
            for col in self.feature_columns:
                if col in data.columns:
                    value = data.iloc[current_index][col]
                    obs_components.append(float(value) if pd.notna(value) else 0.0)

        # Add prediction values for current step
        if self.feature_config.get('use_predictions', False):
            for col in self.prediction_columns:
                if col in data.columns:
                    value = data.iloc[current_index][col]
                    obs_components.append(float(value) if pd.notna(value) else 0.0)

        # Add raw OHLCV data if configured (default: True for backward compatibility)
        if self.feature_config.get('include_raw_prices', True):
            for symbol in self.symbols:
                for col_type in ['Open', 'Close']:
                    col_name = f'{col_type}_{symbol}'
                    if col_name in data.columns:
                        value = data.iloc[current_index][col_name]
                        obs_components.append(float(value) if pd.notna(value) else 0.0)

        # Add positions if configured
        if self.feature_config.get('include_positions', True):
            for symbol in self.symbols:
                obs_components.append(positions.get(symbol, 0.0))

        # Add balance if configured
        if self.feature_config.get('include_balance', True):
            obs_components.append(balance)

        observation = np.array(obs_components, dtype=np.float32)

        # Replace any NaN/Inf with 0
        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)

        return observation

    def get_observation_size(self) -> int:
        """Get the size of the observation vector.

        Returns:
            Number of elements in the observation vector
        """
        size = 0

        # Feature columns
        if self.feature_config.get('use_feature_engineering', False):
            size += len(self.feature_columns)

        # Prediction columns
        if self.feature_config.get('use_predictions', False):
            size += len(self.prediction_columns)

        # Raw OHLCV (Open, Close per symbol) - optional
        if self.feature_config.get('include_raw_prices', True):
            size += len(self.symbols) * 2

        # Positions
        if self.feature_config.get('include_positions', True):
            size += len(self.symbols)

        # Balance
        if self.feature_config.get('include_balance', True):
            size += 1

        return size

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names in the observation.

        Returns:
            List of feature names in order they appear in observation
        """
        names = []

        # Feature columns
        if self.feature_config.get('use_feature_engineering', False):
            names.extend(self.feature_columns)

        # Prediction columns
        if self.feature_config.get('use_predictions', False):
            names.extend(self.prediction_columns)

        # Raw OHLCV - optional
        if self.feature_config.get('include_raw_prices', True):
            for symbol in self.symbols:
                names.extend([f'Open_{symbol}', f'Close_{symbol}'])

        # Positions
        if self.feature_config.get('include_positions', True):
            names.extend([f'position_{symbol}' for symbol in self.symbols])

        # Balance
        if self.feature_config.get('include_balance', True):
            names.append('balance')

        return names
