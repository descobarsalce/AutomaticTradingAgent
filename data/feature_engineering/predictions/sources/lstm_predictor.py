"""
LSTM-based prediction source using TensorFlow/Keras.

Produces multi-output predictions: price, direction, volatility, confidence.
"""
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from data.feature_engineering.predictions.base_prediction import (
    BasePredictionSource,
    PredictionMode,
)

logger = logging.getLogger(__name__)

# TensorFlow import with fallback
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("TensorFlow not available. LSTMPredictor will not function.")


class LSTMPredictor(BasePredictionSource):
    """LSTM-based predictor with multi-head output.

    Produces predictions for:
    - Price change (magnitude)
    - Direction (up/down)
    - Volatility
    - Confidence

    Architecture:
    - Input: Sequence of OHLCV data
    - LSTM layers for temporal patterns
    - Separate output heads for each prediction type and horizon
    """

    def __init__(
        self,
        name: str = "LSTMPredictor",
        sequence_length: int = 20,
        lstm_units: int = 64,
        dense_units: int = 32,
        dropout: float = 0.2,
        horizons: Optional[List[int]] = None,
    ):
        """Initialize LSTM predictor.

        Args:
            name: Source name
            sequence_length: Number of time steps in input sequence
            lstm_units: Number of LSTM units
            dense_units: Number of dense layer units
            dropout: Dropout rate
            horizons: Prediction horizons in days
        """
        super().__init__(name)

        self._sequence_length = sequence_length
        self._lstm_units = lstm_units
        self._dense_units = dense_units
        self._dropout = dropout
        self._horizons = horizons or [1, 5, 21]

        self._model: Optional[Any] = None
        self._scaler_params: Dict[str, Dict[str, float]] = {}
        self._num_features: Optional[int] = None

    def _build_model(self, num_features: int) -> Any:
        """Build the multi-output LSTM model.

        Args:
            num_features: Number of input features

        Returns:
            Compiled Keras model
        """
        if not HAS_TENSORFLOW:
            raise RuntimeError("TensorFlow is required for LSTMPredictor")

        self._num_features = num_features

        # Input layer
        inputs = keras.Input(shape=(self._sequence_length, num_features))

        # LSTM layers
        x = layers.LSTM(self._lstm_units, return_sequences=True)(inputs)
        x = layers.Dropout(self._dropout)(x)
        x = layers.LSTM(self._lstm_units // 2)(x)
        x = layers.Dropout(self._dropout)(x)

        # Shared dense layer
        shared = layers.Dense(self._dense_units, activation='relu')(x)

        # Output heads for each horizon and mode
        outputs = []
        output_names = []

        for horizon in self._horizons:
            # Price prediction (regression)
            price_out = layers.Dense(1, name=f'price_{horizon}d')(shared)
            outputs.append(price_out)
            output_names.append(f'price_{horizon}d')

            # Direction prediction (binary classification)
            direction_out = layers.Dense(
                1, activation='tanh', name=f'direction_{horizon}d'
            )(shared)
            outputs.append(direction_out)
            output_names.append(f'direction_{horizon}d')

            # Volatility prediction (regression, positive)
            vol_out = layers.Dense(
                1, activation='softplus', name=f'volatility_{horizon}d'
            )(shared)
            outputs.append(vol_out)
            output_names.append(f'volatility_{horizon}d')

            # Confidence prediction (0-1)
            conf_out = layers.Dense(
                1, activation='sigmoid', name=f'confidence_{horizon}d'
            )(shared)
            outputs.append(conf_out)
            output_names.append(f'confidence_{horizon}d')

        model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)

        # Compile with appropriate losses
        losses = {}
        for name in output_names:
            if 'direction' in name:
                losses[name] = 'mse'  # Using MSE for tanh output
            else:
                losses[name] = 'mse'

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=losses,
        )

        return model

    def _prepare_training_data(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        start_idx: int,
        end_idx: int,
    ) -> tuple:
        """Prepare training data with sequences and targets.

        Args:
            data: Full DataFrame
            symbols: List of symbols
            start_idx: Start index
            end_idx: End index

        Returns:
            Tuple of (X, y_dict) where y_dict maps output names to targets
        """
        X_sequences = []
        y_targets: Dict[str, List[float]] = {
            f'{mode.value}_{h}d': []
            for h in self._horizons
            for mode in self.modes
        }

        # Compute feature scaling parameters from training window
        self._compute_scaling_params(data, symbols, start_idx, end_idx)

        max_horizon = max(self._horizons)

        for idx in range(start_idx + self._sequence_length, end_idx - max_horizon):
            # Get input sequence
            features = self._prepare_features(data, symbols, idx, self._sequence_length)
            if features is None:
                continue

            # Scale features
            features = self._scale_features(features)

            # Get targets for all horizons (using first symbol for now)
            valid_sample = True
            sample_targets: Dict[str, float] = {}

            for horizon in self._horizons:
                targets = self._compute_targets(data, symbols[0], idx, horizon)
                if targets is None:
                    valid_sample = False
                    break

                for mode in self.modes:
                    key = f'{mode.value}_{horizon}d'
                    sample_targets[key] = targets[mode]

            if valid_sample:
                X_sequences.append(features)
                for key, value in sample_targets.items():
                    y_targets[key].append(value)

        if not X_sequences:
            return None, None

        X = np.array(X_sequences, dtype=np.float32)
        y = {k: np.array(v, dtype=np.float32) for k, v in y_targets.items()}

        return X, y

    def _compute_scaling_params(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        start_idx: int,
        end_idx: int,
    ) -> None:
        """Compute feature scaling parameters."""
        self._scaler_params = {}

        for symbol in symbols:
            for col_type in ['Open', 'High', 'Low', 'Close', 'Volume']:
                col_name = f'{col_type}_{symbol}'
                if col_name in data.columns:
                    col_data = data[col_name].iloc[start_idx:end_idx].dropna()
                    if len(col_data) > 0:
                        self._scaler_params[col_name] = {
                            'mean': float(col_data.mean()),
                            'std': float(col_data.std()) or 1.0,
                        }

    def _scale_features(self, features: np.ndarray) -> np.ndarray:
        """Apply z-score scaling to features."""
        if not self._scaler_params:
            return features

        scaled = features.copy()
        # Apply scaling column by column
        num_cols = scaled.shape[1] if len(scaled.shape) > 1 else 1
        col_idx = 0

        for col_name, params in self._scaler_params.items():
            if col_idx < num_cols:
                if len(scaled.shape) > 1:
                    scaled[:, col_idx] = (
                        scaled[:, col_idx] - params['mean']
                    ) / params['std']
                col_idx += 1

        return np.clip(scaled, -10, 10)

    def train(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        start_idx: int,
        end_idx: int,
    ) -> None:
        """Train the LSTM model.

        Args:
            data: Full DataFrame with OHLCV data
            symbols: List of symbols
            start_idx: Start index
            end_idx: End index
        """
        if not HAS_TENSORFLOW:
            raise RuntimeError("TensorFlow is required for LSTMPredictor")

        logger.info(f"Training {self.name} on {end_idx - start_idx} samples")

        # Prepare training data
        X, y = self._prepare_training_data(data, symbols, start_idx, end_idx)

        if X is None or len(X) < 10:
            logger.warning("Insufficient training data")
            return

        # Build model if needed
        num_features = X.shape[2]
        if self._model is None or self._num_features != num_features:
            self._model = self._build_model(num_features)

        # Train
        self._model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=5,
                    restore_best_weights=True,
                ),
            ],
        )

        self._is_trained = True
        logger.info(f"{self.name} training complete")

    def predict(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        current_idx: int,
    ) -> Dict[str, Dict[int, Dict[PredictionMode, float]]]:
        """Generate predictions for current time step.

        Args:
            data: Full DataFrame
            symbols: List of symbols
            current_idx: Current time index

        Returns:
            Nested dict: {symbol: {horizon: {mode: value}}}
        """
        if not self._is_trained or self._model is None:
            logger.warning(f"{self.name} is not trained")
            return {}

        # Prepare input sequence
        features = self._prepare_features(
            data, symbols, current_idx, self._sequence_length
        )
        if features is None:
            return {}

        # Scale and reshape for prediction
        features = self._scale_features(features)
        X = np.expand_dims(features, axis=0)

        # Get predictions
        predictions = self._model.predict(X, verbose=0)

        # Parse predictions into structured format
        result: Dict[str, Dict[int, Dict[PredictionMode, float]]] = {}

        pred_idx = 0
        for horizon in self._horizons:
            for mode in self.modes:
                value = float(predictions[pred_idx][0][0])
                pred_idx += 1

                # Store for each symbol (same prediction for now)
                for symbol in symbols:
                    if symbol not in result:
                        result[symbol] = {}
                    if horizon not in result[symbol]:
                        result[symbol][horizon] = {}
                    result[symbol][horizon][mode] = value

        return result

    def save_model(self, path: str) -> None:
        """Save model and scaler to disk.

        Args:
            path: Directory path
        """
        if not HAS_TENSORFLOW:
            return

        os.makedirs(path, exist_ok=True)

        if self._model is not None:
            model_path = os.path.join(path, 'model.keras')
            self._model.save(model_path)

        # Save scaler params
        import json
        scaler_path = os.path.join(path, 'scaler_params.json')
        with open(scaler_path, 'w') as f:
            json.dump(self._scaler_params, f)

        # Save config
        config_path = os.path.join(path, 'config.json')
        config = {
            'sequence_length': self._sequence_length,
            'lstm_units': self._lstm_units,
            'dense_units': self._dense_units,
            'dropout': self._dropout,
            'horizons': self._horizons,
            'num_features': self._num_features,
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)

        logger.info(f"Saved {self.name} to {path}")

    def load_model(self, path: str) -> bool:
        """Load model and scaler from disk.

        Args:
            path: Directory path

        Returns:
            True if loaded successfully
        """
        if not HAS_TENSORFLOW:
            return False

        model_path = os.path.join(path, 'model.keras')
        scaler_path = os.path.join(path, 'scaler_params.json')
        config_path = os.path.join(path, 'config.json')

        if not os.path.exists(model_path):
            return False

        try:
            import json

            # Load config
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self._sequence_length = config.get('sequence_length', 20)
                    self._lstm_units = config.get('lstm_units', 64)
                    self._dense_units = config.get('dense_units', 32)
                    self._dropout = config.get('dropout', 0.2)
                    self._horizons = config.get('horizons', [1, 5, 21])
                    self._num_features = config.get('num_features')

            # Load scaler
            if os.path.exists(scaler_path):
                with open(scaler_path, 'r') as f:
                    self._scaler_params = json.load(f)

            # Load model
            self._model = keras.models.load_model(model_path)
            self._is_trained = True

            logger.info(f"Loaded {self.name} from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
