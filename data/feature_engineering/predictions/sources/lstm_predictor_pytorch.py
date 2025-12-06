"""
LSTM-based prediction source using PyTorch.

Produces multi-output predictions: price, direction, volatility, confidence.
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.feature_engineering.predictions.base_prediction import (
    BasePredictionSource,
    PredictionMode,
)

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """Multi-output LSTM model for time series prediction."""

    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        lstm_units: int = 64,
        dense_units: int = 32,
        dropout: float = 0.2,
        horizons: List[int] = None,
    ):
        super().__init__()
        horizons = horizons or [1, 5, 21]

        self.lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_units,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=lstm_units,
            hidden_size=lstm_units // 2,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.shared_dense = nn.Linear(lstm_units // 2, dense_units)
        self.relu = nn.ReLU()

        # Output heads for each horizon and mode
        # Modes: price, direction, volatility, confidence
        self.output_heads = nn.ModuleDict()
        for horizon in horizons:
            self.output_heads[f'price_{horizon}d'] = nn.Linear(dense_units, 1)
            self.output_heads[f'direction_{horizon}d'] = nn.Sequential(
                nn.Linear(dense_units, 1),
                nn.Tanh()
            )
            self.output_heads[f'volatility_{horizon}d'] = nn.Sequential(
                nn.Linear(dense_units, 1),
                nn.Softplus()
            )
            self.output_heads[f'confidence_{horizon}d'] = nn.Sequential(
                nn.Linear(dense_units, 1),
                nn.Sigmoid()
            )

        self.horizons = horizons

    def forward(self, x):
        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # Take the last time step
        x = x[:, -1, :]

        # Shared dense layer
        x = self.relu(self.shared_dense(x))

        # Output heads
        outputs = {}
        for name, head in self.output_heads.items():
            outputs[name] = head(x)

        return outputs


class LSTMPredictorPyTorch(BasePredictionSource):
    """LSTM-based predictor with multi-head output using PyTorch.

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

        self._model: Optional[LSTMModel] = None
        self._scaler_params: Dict[str, Dict[str, float]] = {}
        self._num_features: Optional[int] = None
        self._device = torch.device('cpu')

    def _build_model(self, num_features: int) -> LSTMModel:
        """Build the multi-output LSTM model.

        Args:
            num_features: Number of input features

        Returns:
            LSTMModel instance
        """
        self._num_features = num_features

        model = LSTMModel(
            num_features=num_features,
            sequence_length=self._sequence_length,
            lstm_units=self._lstm_units,
            dense_units=self._dense_units,
            dropout=self._dropout,
            horizons=self._horizons,
        )

        return model.to(self._device)

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

        loop_start = start_idx + self._sequence_length
        loop_end = end_idx - max_horizon
        logger.info(
            f"Preparing training data: loop from {loop_start} to {loop_end} "
            f"(sequence_length={self._sequence_length}, max_horizon={max_horizon})"
        )

        features_none_count = 0
        targets_none_count = 0

        for idx in range(loop_start, loop_end):
            # Get input sequence
            features = self._prepare_features(data, symbols, idx, self._sequence_length)
            if features is None:
                features_none_count += 1
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
                    targets_none_count += 1
                    break

                for mode in self.modes:
                    key = f'{mode.value}_{horizon}d'
                    sample_targets[key] = targets[mode]

            if valid_sample:
                X_sequences.append(features)
                for key, value in sample_targets.items():
                    y_targets[key].append(value)

        logger.info(
            f"Training data prep complete: {len(X_sequences)} samples, "
            f"{features_none_count} features_none, {targets_none_count} targets_none"
        )

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
                    std = params['std'] if params['std'] != 0 else 1.0
                    scaled[:, col_idx] = (
                        scaled[:, col_idx] - params['mean']
                    ) / std
                col_idx += 1

        # Handle NaN values and clip
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=10.0, neginf=-10.0)
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
        logger.info(f"Training {self.name} on {end_idx - start_idx} samples")

        # Prepare training data
        X, y = self._prepare_training_data(data, symbols, start_idx, end_idx)

        if X is None or len(X) < 10:
            # Debug: check why training data is insufficient
            sample_count = len(X) if X is not None else 0
            logger.warning(
                f"Insufficient training data: got {sample_count} samples (need >= 10). "
                f"Data columns: {list(data.columns)[:5]}..., "
                f"Expected columns like: Open_{symbols[0]}, Close_{symbols[0]}. "
                f"Index range: {start_idx} to {end_idx}"
            )
            return

        # Build model if needed
        num_features = X.shape[2]
        if self._model is None or self._num_features != num_features:
            self._model = self._build_model(num_features)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self._device)
        y_tensors = {k: torch.FloatTensor(v).unsqueeze(1).to(self._device) for k, v in y.items()}

        # Split into train/val
        val_size = int(len(X_tensor) * 0.2)
        train_size = len(X_tensor) - val_size

        # Create data loaders
        train_dataset = TensorDataset(X_tensor[:train_size], *[y_tensors[k][:train_size] for k in sorted(y_tensors.keys())])
        val_dataset = TensorDataset(X_tensor[train_size:], *[y_tensors[k][train_size:] for k in sorted(y_tensors.keys())])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # Training setup
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Early stopping
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_model_state = None

        target_keys = sorted(y_tensors.keys())

        try:
            logger.info(f"Starting model training with X shape {X.shape}")
            self._model.train()

            for epoch in range(50):
                # Training phase
                train_loss = 0.0
                for batch in train_loader:
                    X_batch = batch[0]
                    y_batch = {k: batch[i+1] for i, k in enumerate(target_keys)}

                    optimizer.zero_grad()
                    outputs = self._model(X_batch)

                    loss = sum(criterion(outputs[k], y_batch[k]) for k in target_keys)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # Validation phase
                self._model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        X_batch = batch[0]
                        y_batch = {k: batch[i+1] for i, k in enumerate(target_keys)}

                        outputs = self._model(X_batch)
                        loss = sum(criterion(outputs[k], y_batch[k]) for k in target_keys)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                self._model.train()

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self._model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

                if epoch % 10 == 0:
                    logger.debug(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # Restore best model
            if best_model_state is not None:
                self._model.load_state_dict(best_model_state)

            self._is_trained = True
            logger.info(f"{self.name} training complete")

        except Exception as e:
            logger.error(f"{self.name} training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

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
        X = torch.FloatTensor(features).unsqueeze(0).to(self._device)

        # Get predictions
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(X)

        # Parse predictions into structured format
        result: Dict[str, Dict[int, Dict[PredictionMode, float]]] = {}

        for horizon in self._horizons:
            for mode in self.modes:
                key = f'{mode.value}_{horizon}d'
                value = float(outputs[key][0][0].cpu().numpy())

                # Handle NaN values
                if np.isnan(value) or np.isinf(value):
                    value = 0.0

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
        os.makedirs(path, exist_ok=True)

        if self._model is not None:
            model_path = os.path.join(path, 'model.pt')
            torch.save(self._model.state_dict(), model_path)

        # Save scaler params
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
            'backend': 'pytorch',
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
        model_path = os.path.join(path, 'model.pt')
        scaler_path = os.path.join(path, 'scaler_params.json')
        config_path = os.path.join(path, 'config.json')

        if not os.path.exists(model_path):
            return False

        try:
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

            # Build and load model
            if self._num_features is not None:
                self._model = self._build_model(self._num_features)
                self._model.load_state_dict(torch.load(model_path, map_location=self._device))
                self._model.eval()
                self._is_trained = True

            logger.info(f"Loaded {self.name} from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
