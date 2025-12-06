"""
ML Feature Models Tab
Allows users to train, validate, and manage ML prediction models
that generate features for the main RL trading agent.
"""
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# Model type configurations (using PyTorch backend)
MODEL_TYPES = {
    'LSTM': {
        'class': 'LSTMPredictorPyTorch',
        'module': 'data.feature_engineering.predictions.sources.lstm_predictor_pytorch',
        'params': {
            'sequence_length': {'min': 5, 'max': 100, 'default': 20, 'help': 'Input sequence length'},
            'lstm_units': {'min': 16, 'max': 256, 'default': 64, 'help': 'LSTM layer units'},
            'dense_units': {'min': 8, 'max': 128, 'default': 32, 'help': 'Dense layer units'},
            'dropout': {'min': 0.0, 'max': 0.5, 'default': 0.2, 'step': 0.05, 'help': 'Dropout rate'},
        }
    },
    'Transformer': {
        'class': 'TransformerPredictorPyTorch',
        'module': 'data.feature_engineering.predictions.sources.transformer_predictor_pytorch',
        'params': {
            'sequence_length': {'min': 5, 'max': 100, 'default': 20, 'help': 'Input sequence length'},
            'd_model': {'min': 16, 'max': 256, 'default': 64, 'help': 'Model dimension'},
            'num_heads': {'min': 1, 'max': 8, 'default': 4, 'help': 'Attention heads'},
            'num_layers': {'min': 1, 'max': 6, 'default': 2, 'help': 'Encoder layers'},
            'ff_dim': {'min': 32, 'max': 512, 'default': 128, 'help': 'Feed-forward dimension'},
            'dropout': {'min': 0.0, 'max': 0.5, 'default': 0.1, 'step': 0.05, 'help': 'Dropout rate'},
        }
    },
}

# Default horizons for predictions
DEFAULT_HORIZONS = [1, 5, 21]

# Model storage directory
MODELS_DIR = '.prediction_models'


def initialize_ml_models_state():
    """Initialize session state for ML models tab."""
    if 'ml_models_config' not in st.session_state:
        st.session_state.ml_models_config = {
            'registered_models': {},  # {model_id: {type, symbol, config, trained, last_trained, metrics}}
            'selected_for_features': [],  # List of model_ids enabled for feature generation
            'validation_results': {},  # {model_id: WalkForwardResults}
        }

    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load any saved models from disk
    _load_saved_models()


def _load_saved_models():
    """Load model metadata from disk."""
    if not os.path.exists(MODELS_DIR):
        return

    config = st.session_state.ml_models_config

    for model_dir in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, model_dir)
        config_file = os.path.join(model_path, 'model_info.json')

        if os.path.isdir(model_path) and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    model_info = json.load(f)
                    model_id = model_info.get('model_id', model_dir)
                    if model_id not in config['registered_models']:
                        config['registered_models'][model_id] = model_info
            except Exception as e:
                logger.debug(f"Could not load model info from {config_file}: {e}")


def _save_model_info(model_id: str, model_info: Dict[str, Any]):
    """Save model metadata to disk."""
    model_path = os.path.join(MODELS_DIR, model_id)
    os.makedirs(model_path, exist_ok=True)

    config_file = os.path.join(model_path, 'model_info.json')
    with open(config_file, 'w') as f:
        json.dump(model_info, f, indent=2, default=str)


def get_available_stocks() -> List[str]:
    """Get list of stocks available in the database."""
    try:
        if 'data_handler' not in st.session_state:
            return []
        from sqlalchemy import distinct
        from data.database import StockData
        symbols = [row[0] for row in st.session_state.data_handler.query(distinct(StockData.symbol)).all()]
        return sorted(symbols)
    except Exception as e:
        logger.error(f"Error getting available stocks: {e}")
        return []


def _get_stock_data(symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """Fetch stock data from database."""
    try:
        if 'data_handler' not in st.session_state:
            return None

        # Fetch extra data for sequence building
        fetch_start = start_date - timedelta(days=60)
        data = st.session_state.data_handler.fetch_data(
            [symbol], fetch_start, end_date, use_SQL=True
        )
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None


def _create_predictor(model_type: str, symbol: str, params: Dict[str, Any]):
    """Create a predictor instance using PyTorch backend."""
    model_config = MODEL_TYPES.get(model_type)
    if not model_config:
        raise ValueError(f"Unknown model type: {model_type}")

    model_id = f"{model_type}_{symbol}"

    if model_type == 'LSTM':
        from data.feature_engineering.predictions.sources.lstm_predictor_pytorch import LSTMPredictorPyTorch
        predictor = LSTMPredictorPyTorch(
            name=model_id,
            sequence_length=params.get('sequence_length', 20),
            lstm_units=params.get('lstm_units', 64),
            dense_units=params.get('dense_units', 32),
            dropout=params.get('dropout', 0.2),
            horizons=params.get('horizons', DEFAULT_HORIZONS),
        )
    elif model_type == 'Transformer':
        from data.feature_engineering.predictions.sources.transformer_predictor_pytorch import TransformerPredictorPyTorch
        predictor = TransformerPredictorPyTorch(
            name=model_id,
            sequence_length=params.get('sequence_length', 20),
            d_model=params.get('d_model', 64),
            num_heads=params.get('num_heads', 4),
            num_layers=params.get('num_layers', 2),
            ff_dim=params.get('ff_dim', 128),
            dropout=params.get('dropout', 0.1),
            horizons=params.get('horizons', DEFAULT_HORIZONS),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return predictor


def _run_walk_forward_validation(
    predictor,
    data: pd.DataFrame,
    symbol: str,
    train_window: int,
    test_window: int,
    step_size: int,
    progress_bar,
    status_container=None,
):
    """Run walk-forward validation with progress tracking."""
    from data.feature_engineering.predictions.walk_forward_validator import WalkForwardValidator

    validator = WalkForwardValidator(
        train_window_days=train_window,
        test_window_days=test_window,
        step_days=step_size,
    )

    # Calculate expected folds
    expected_folds = validator._generate_folds(len(data))
    if status_container:
        status_container.info(f"Starting validation with {len(expected_folds)} expected folds...")

    def progress_callback(fold_idx: int, total_folds: int):
        progress = (fold_idx + 1) / total_folds
        progress_bar.progress(progress, text=f"Fold {fold_idx + 1}/{total_folds}")

    results = validator.validate(
        predictor=predictor,
        data=data,
        symbol=symbol,
        progress_callback=progress_callback,
    )

    return results


def display_model_configuration():
    """Display model configuration section."""
    st.subheader("Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        model_type = st.selectbox(
            "Model Type",
            options=list(MODEL_TYPES.keys()),
            key="ml_model_type",
            help="Select the type of prediction model"
        )

        available_stocks = get_available_stocks()
        if not available_stocks:
            st.warning("No stocks in database. Add data in Database Explorer first.")
            return None, None, None

        symbol = st.selectbox(
            "Stock Symbol",
            options=available_stocks,
            key="ml_model_symbol",
            help="Select stock for per-stock model"
        )

    with col2:
        st.markdown("**Hyperparameters**")
        params = {}
        model_config = MODEL_TYPES[model_type]

        for param_name, param_config in model_config['params'].items():
            if 'step' in param_config:
                # Float parameter
                params[param_name] = st.slider(
                    param_name.replace('_', ' ').title(),
                    min_value=float(param_config['min']),
                    max_value=float(param_config['max']),
                    value=float(param_config['default']),
                    step=param_config.get('step', 0.01),
                    key=f"ml_param_{param_name}",
                    help=param_config.get('help', ''),
                )
            else:
                # Integer parameter
                params[param_name] = st.slider(
                    param_name.replace('_', ' ').title(),
                    min_value=param_config['min'],
                    max_value=param_config['max'],
                    value=param_config['default'],
                    key=f"ml_param_{param_name}",
                    help=param_config.get('help', ''),
                )

    # Horizons selection
    horizons = st.multiselect(
        "Prediction Horizons (days)",
        options=[1, 5, 10, 21, 63],
        default=DEFAULT_HORIZONS,
        key="ml_horizons",
        help="Days ahead to predict"
    )
    params['horizons'] = horizons

    return model_type, symbol, params


def display_validation_settings():
    """Display walk-forward validation settings."""
    st.subheader("Walk-Forward Validation")

    col1, col2, col3 = st.columns(3)

    with col1:
        train_window = st.number_input(
            "Train Window (days)",
            min_value=30,
            max_value=1000,
            value=252,
            step=21,
            key="ml_train_window",
            help="Number of days for training window"
        )

    with col2:
        test_window = st.number_input(
            "Test Window (days)",
            min_value=5,
            max_value=126,
            value=21,
            step=5,
            key="ml_test_window",
            help="Number of days for testing window"
        )

    with col3:
        step_size = st.number_input(
            "Step Size (days)",
            min_value=1,
            max_value=63,
            value=21,
            step=1,
            key="ml_step_size",
            help="Days to step forward between folds"
        )

    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=756),  # ~3 years
            key="ml_start_date",
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime(2025, 10, 1),
            key="ml_end_date",
        )

    return train_window, test_window, step_size, start_date, end_date


def display_registered_models():
    """Display registered models table."""
    st.subheader("Registered Models")

    config = st.session_state.ml_models_config
    models = config['registered_models']

    if not models:
        st.info("No models registered yet. Train a model above.")
        return

    # Build table data
    table_data = []
    for model_id, info in models.items():
        table_data.append({
            'Model ID': model_id,
            'Type': info.get('type', 'Unknown'),
            'Symbol': info.get('symbol', 'Unknown'),
            'Status': 'Trained' if info.get('trained', False) else 'Pending',
            'Last Trained': info.get('last_trained', '-'),
            'MAE (1d)': f"{info.get('metrics', {}).get('mae_1d', 0):.4f}" if info.get('metrics') else '-',
            'Dir Acc (1d)': f"{info.get('metrics', {}).get('dir_acc_1d', 0):.1%}" if info.get('metrics') else '-',
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Model actions
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_model = st.selectbox(
            "Select Model",
            options=list(models.keys()),
            key="ml_selected_model",
        )

    with col2:
        if st.button("Delete Selected", key="ml_delete"):
            if selected_model in config['registered_models']:
                del config['registered_models'][selected_model]
                # Also remove from disk
                model_path = os.path.join(MODELS_DIR, selected_model)
                if os.path.exists(model_path):
                    import shutil
                    shutil.rmtree(model_path)
                st.success(f"Deleted {selected_model}")
                st.rerun()

    with col3:
        is_enabled = selected_model in config.get('selected_for_features', [])
        if st.button(
            "Disable for Features" if is_enabled else "Enable for Features",
            key="ml_toggle_features",
        ):
            if is_enabled:
                config['selected_for_features'].remove(selected_model)
                st.success(f"Disabled {selected_model} for feature generation")
            else:
                config['selected_for_features'].append(selected_model)
                st.success(f"Enabled {selected_model} for feature generation")
            st.rerun()


def display_validation_results(model_id: str):
    """Display validation results for a model."""
    config = st.session_state.ml_models_config
    results = config.get('validation_results', {}).get(model_id)

    if not results:
        return

    st.subheader(f"Validation Results: {model_id}")

    # Aggregated metrics
    st.markdown("**Aggregated Metrics**")
    metrics_cols = st.columns(4)

    horizons = sorted(results.mae_by_horizon.keys())
    if horizons:
        h = horizons[0]  # Show primary horizon
        with metrics_cols[0]:
            st.metric("MAE", f"{results.mae_by_horizon.get(h, 0):.4f}")
        with metrics_cols[1]:
            st.metric("RMSE", f"{results.rmse_by_horizon.get(h, 0):.4f}")
        with metrics_cols[2]:
            st.metric("Direction Accuracy", f"{results.direction_accuracy_by_horizon.get(h, 0):.1%}")
        with metrics_cols[3]:
            st.metric("Folds", results.num_folds)

    # Per-fold results
    with st.expander("Per-Fold Results", expanded=False):
        fold_data = []
        for fold in results.fold_results:
            fold_data.append({
                'Fold': fold.fold_idx + 1,
                'Train Period': f"{fold.train_start_date} to {fold.train_end_date}",
                'Test Period': f"{fold.test_start_date} to {fold.test_end_date}",
                'MAE (1d)': f"{fold.mae_by_horizon.get(1, 0):.4f}",
                'Dir Acc (1d)': f"{fold.direction_accuracy_by_horizon.get(1, 0):.1%}",
            })

        if fold_data:
            st.dataframe(pd.DataFrame(fold_data), use_container_width=True, hide_index=True)

    # Prediction chart
    with st.expander("Prediction vs Actual Chart", expanded=False):
        if results.all_predictions and results.all_actuals:
            import plotly.graph_objects as go

            h = horizons[0] if horizons else 1
            preds = results.all_predictions.get(h, [])
            actuals = results.all_actuals.get(h, [])

            if preds and actuals:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=actuals,
                    mode='lines',
                    name='Actual',
                    line=dict(color='blue', width=1),
                ))
                fig.add_trace(go.Scatter(
                    y=preds,
                    mode='lines',
                    name='Predicted',
                    line=dict(color='red', width=1, dash='dot'),
                ))
                fig.update_layout(
                    title=f'{h}-Day Price Change: Predicted vs Actual',
                    xaxis_title='Sample',
                    yaxis_title='Price Change',
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True, key=f"ml_chart_{model_id}")


def display_ml_models_tab():
    """Renders the ML Feature Models tab."""
    st.header("ML Feature Models")
    st.caption("Train prediction models to generate features for the RL trading agent")

    initialize_ml_models_state()

    # Model Configuration
    result = display_model_configuration()
    if result[0] is None:
        return

    model_type, symbol, params = result

    st.divider()

    # Validation Settings
    train_window, test_window, step_size, start_date, end_date = display_validation_settings()

    # Train button
    st.divider()

    if st.button("Train & Validate", type="primary", key="ml_train_btn"):
        if start_date >= end_date:
            st.error("Start date must be before end date")
            return

        model_id = f"{model_type}_{symbol}"

        with st.spinner(f"Fetching data for {symbol}..."):
            data = _get_stock_data(
                symbol,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.min.time()),
            )

        if data is None or data.empty:
            st.error(f"No data available for {symbol} in the selected range")
            return

        # Verify data has required columns
        required_cols = [f'Open_{symbol}', f'Close_{symbol}', f'High_{symbol}', f'Low_{symbol}', f'Volume_{symbol}']
        missing_cols = [col for col in required_cols if col not in data.columns]

        if missing_cols:
            st.error(f"Data missing required columns: {missing_cols}")
            st.info(f"Available columns: {list(data.columns)[:10]}...")
            return

        st.info(f"Loaded {len(data)} data points for {symbol} with columns: {required_cols[:3]}...")

        # Create predictor
        try:
            predictor = _create_predictor(model_type, symbol, params)
            if predictor is None:
                return
        except Exception as e:
            st.error(f"Failed to create predictor: {e}")
            return

        # Run walk-forward validation
        progress_bar = st.progress(0, text="Starting validation...")
        status_container = st.empty()
        error_container = st.empty()

        try:
            # Set up logging to capture errors
            import io
            import sys

            # Capture stderr to show TensorFlow errors
            old_stderr = sys.stderr
            sys.stderr = captured_stderr = io.StringIO()

            results = _run_walk_forward_validation(
                predictor=predictor,
                data=data,
                symbol=symbol,
                train_window=train_window,
                test_window=test_window,
                step_size=step_size,
                progress_bar=progress_bar,
                status_container=status_container,
            )

            # Restore stderr and check for errors
            sys.stderr = old_stderr
            stderr_output = captured_stderr.getvalue()
            if stderr_output and 'error' in stderr_output.lower():
                with error_container:
                    st.warning("Training produced warnings/errors (see terminal for details)")

            progress_bar.progress(1.0, text="Validation complete!")

            if results.num_folds == 0:
                st.warning("No validation folds could be completed.")
                st.info(
                    f"Debug info: Data has {len(data)} rows. "
                    f"Attempted {results.total_folds_attempted} folds. "
                    f"Training successes: {results.training_successes}, failures: {results.training_failures}. "
                    f"With train_window={train_window}, test_window={test_window}, "
                    f"minimum data needed is {train_window + test_window} rows."
                )
                st.warning(
                    "All training attempts failed. This usually means:\n"
                    "1. Not enough training samples (try smaller train_window or sequence_length)\n"
                    "2. TensorFlow error during model training (check terminal for details)"
                )
                # Show any stderr output
                if stderr_output:
                    with st.expander("Error Details", expanded=True):
                        st.code(stderr_output[-2000:])  # Last 2000 chars
                return

            # Save model
            model_path = os.path.join(MODELS_DIR, model_id)
            predictor.save_model(model_path)

            # Store results
            config = st.session_state.ml_models_config

            model_info = {
                'model_id': model_id,
                'type': model_type,
                'symbol': symbol,
                'params': params,
                'trained': True,
                'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'metrics': {
                    'mae_1d': results.mae_by_horizon.get(1, 0),
                    'rmse_1d': results.rmse_by_horizon.get(1, 0),
                    'dir_acc_1d': results.direction_accuracy_by_horizon.get(1, 0),
                    'num_folds': results.num_folds,
                },
            }

            config['registered_models'][model_id] = model_info
            config['validation_results'][model_id] = results

            # Save to disk
            _save_model_info(model_id, model_info)

            st.success(f"Model {model_id} trained successfully!")

            # Display results
            display_validation_results(model_id)

        except Exception as e:
            # Restore stderr
            sys.stderr = old_stderr
            stderr_output = captured_stderr.getvalue()

            logger.error(f"Validation failed: {e}", exc_info=True)
            st.error(f"Validation failed: {e}")

            # Show detailed error
            if stderr_output:
                with st.expander("Error Details", expanded=True):
                    st.code(stderr_output[-2000:])

    st.divider()

    # Registered Models
    display_registered_models()

    # Show results for selected model
    config = st.session_state.ml_models_config
    selected = st.session_state.get('ml_selected_model')
    if selected and selected in config.get('validation_results', {}):
        display_validation_results(selected)


def get_enabled_prediction_models() -> List[str]:
    """Get list of model IDs enabled for feature generation."""
    initialize_ml_models_state()
    return st.session_state.ml_models_config.get('selected_for_features', [])


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """Get info for a specific model."""
    initialize_ml_models_state()
    return st.session_state.ml_models_config.get('registered_models', {}).get(model_id)
