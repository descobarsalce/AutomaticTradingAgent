"""
Debug script to identify why ML model training fails.
"""
import logging
import sys

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

import numpy as np
import pandas as pd

# Import our prediction classes
from data.feature_engineering.predictions.sources.lstm_predictor import LSTMPredictor
from data.feature_engineering.predictions.base_prediction import PredictionMode

def create_test_data(symbol: str, num_days: int = 300) -> pd.DataFrame:
    """Create synthetic test data with correct column format."""
    np.random.seed(42)

    dates = pd.date_range(start='2022-01-01', periods=num_days, freq='D')

    # Create realistic price data
    base_price = 100
    returns = np.random.randn(num_days) * 0.02  # 2% daily volatility
    prices = base_price * np.cumprod(1 + returns)

    data = pd.DataFrame({
        f'Open_{symbol}': prices * (1 + np.random.randn(num_days) * 0.005),
        f'High_{symbol}': prices * (1 + np.abs(np.random.randn(num_days) * 0.01)),
        f'Low_{symbol}': prices * (1 - np.abs(np.random.randn(num_days) * 0.01)),
        f'Close_{symbol}': prices,
        f'Volume_{symbol}': np.random.randint(1000000, 10000000, num_days),
    }, index=dates)

    return data


def test_prepare_features():
    """Test the _prepare_features method."""
    print("\n" + "="*60)
    print("TEST 1: _prepare_features")
    print("="*60)

    symbol = 'AAPL'
    data = create_test_data(symbol, 300)
    print(f"Created test data: {len(data)} rows")
    print(f"Columns: {list(data.columns)}")
    print(f"Index type: {type(data.index)}")
    print(f"Sample data:\n{data.head()}")

    # Create predictor
    predictor = LSTMPredictor(
        name=f"LSTM_{symbol}",
        sequence_length=20,
        lstm_units=64,
        horizons=[1, 5, 21]
    )

    # Test _prepare_features at various indices
    test_indices = [20, 50, 100, 150]
    for idx in test_indices:
        features = predictor._prepare_features(data, [symbol], idx, 20)
        if features is None:
            print(f"  idx={idx}: features is None!")
        else:
            print(f"  idx={idx}: features shape={features.shape}, min={features.min():.4f}, max={features.max():.4f}")

    return data, predictor


def test_compute_targets():
    """Test the _compute_targets method."""
    print("\n" + "="*60)
    print("TEST 2: _compute_targets")
    print("="*60)

    symbol = 'AAPL'
    data = create_test_data(symbol, 300)
    predictor = LSTMPredictor(name=f"LSTM_{symbol}", sequence_length=20, horizons=[1, 5, 21])

    test_indices = [50, 100, 150, 200, 250]
    for idx in test_indices:
        for horizon in [1, 5, 21]:
            targets = predictor._compute_targets(data, symbol, idx, horizon)
            if targets is None:
                print(f"  idx={idx}, horizon={horizon}: targets is None!")
            else:
                print(f"  idx={idx}, horizon={horizon}: price={targets[PredictionMode.PRICE]:.4f}, dir={targets[PredictionMode.DIRECTION]:.0f}")


def test_prepare_training_data():
    """Test the _prepare_training_data method."""
    print("\n" + "="*60)
    print("TEST 3: _prepare_training_data")
    print("="*60)

    symbol = 'AAPL'
    data = create_test_data(symbol, 300)
    predictor = LSTMPredictor(name=f"LSTM_{symbol}", sequence_length=20, horizons=[1, 5, 21])

    start_idx = 0
    end_idx = 252  # 1 year

    print(f"Calling _prepare_training_data with start_idx={start_idx}, end_idx={end_idx}")

    X, y = predictor._prepare_training_data(data, [symbol], start_idx, end_idx)

    if X is None:
        print("  X is None - training data preparation failed!")
    else:
        print(f"  X shape: {X.shape}")
        print(f"  y keys: {list(y.keys())}")
        print(f"  Sample counts per target: {[(k, len(v)) for k, v in y.items()][:3]}")


def test_full_training():
    """Test full training cycle."""
    print("\n" + "="*60)
    print("TEST 4: Full Training")
    print("="*60)

    symbol = 'AAPL'
    data = create_test_data(symbol, 300)
    predictor = LSTMPredictor(name=f"LSTM_{symbol}", sequence_length=20, horizons=[1, 5, 21])

    print(f"Training on data[0:252]...")

    try:
        predictor.train(data, [symbol], 0, 252)
        print(f"  is_trained: {predictor.is_trained}")

        if predictor.is_trained:
            print("  Training succeeded!")
            # Test prediction
            preds = predictor.predict(data, [symbol], 260)
            if preds:
                print(f"  Prediction at idx 260: {preds}")
            else:
                print("  Prediction returned empty dict")
    except Exception as e:
        print(f"  Training failed with error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def test_walk_forward_validation():
    """Test walk-forward validation."""
    print("\n" + "="*60)
    print("TEST 5: Walk-Forward Validation")
    print("="*60)

    from data.feature_engineering.predictions.walk_forward_validator import WalkForwardValidator

    symbol = 'AAPL'
    data = create_test_data(symbol, 600)  # More data for validation
    predictor = LSTMPredictor(name=f"LSTM_{symbol}", sequence_length=20, horizons=[1, 5, 21])

    validator = WalkForwardValidator(
        train_window_days=252,
        test_window_days=21,
        step_days=21,
    )

    print(f"Running walk-forward validation with {len(data)} data points...")

    # Check expected folds
    folds = validator._generate_folds(len(data))
    print(f"  Expected folds: {len(folds)}")
    if folds:
        print(f"  First fold: train[{folds[0][0]}:{folds[0][1]}], test[{folds[0][2]}:{folds[0][3]}]")

    results = validator.validate(predictor, data, symbol)

    print(f"  Folds attempted: {results.total_folds_attempted}")
    print(f"  Training successes: {results.training_successes}")
    print(f"  Training failures: {results.training_failures}")
    print(f"  Completed folds: {results.num_folds}")

    if results.num_folds > 0:
        print(f"  MAE (1d): {results.mae_by_horizon.get(1, 'N/A')}")
        print(f"  Direction Accuracy (1d): {results.direction_accuracy_by_horizon.get(1, 'N/A')}")


if __name__ == '__main__':
    print("="*60)
    print("ML MODEL TRAINING DIAGNOSTICS")
    print("="*60)

    test_prepare_features()
    test_compute_targets()
    test_prepare_training_data()
    test_full_training()
    test_walk_forward_validation()

    print("\n" + "="*60)
    print("DIAGNOSTICS COMPLETE")
    print("="*60)
