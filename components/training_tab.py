"""
Training Interface Component
Handles the model training interface
"""
import streamlit as st
import logging
import os

from core.training_functions import run_training
from core.hyperparameter_search import load_best_params

logger = logging.getLogger(__name__)


def display_training_tab():
    """Renders the compact model training interface tab."""
    st.header("Model Training")

    # Compact config summary - all in one row
    stocks = st.session_state.get('stock_names', ['AAPL', 'MSFT'])
    env_params = st.session_state.get('env_params', {})
    train_start = st.session_state.get('train_start_date')
    train_end = st.session_state.get('train_end_date')

    start_str = train_start.strftime('%Y-%m-%d') if hasattr(train_start, 'strftime') else 'Not set'
    end_str = train_end.strftime('%Y-%m-%d') if hasattr(train_end, 'strftime') else 'Not set'

    # Compact config display
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.caption(f"**Stocks:** {', '.join(stocks[:3])}{'...' if len(stocks) > 3 else ''}")
    with c2:
        st.caption(f"**Balance:** ${env_params.get('initial_balance', 10000):,.0f}")
    with c3:
        st.caption(f"**Period:** {start_str} to {end_str}")
    with c4:
        enable_logging = st.checkbox("Logging", value=False, key="training_logging")
        if enable_logging:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.CRITICAL)

    # Load best params
    best_params = load_best_params()

    # PPO params summary (if available)
    if best_params:
        ppo = best_params.get('params', {})
        st.caption(
            f"**PPO:** LR={ppo.get('learning_rate', 0):.2e} | "
            f"Steps={ppo.get('n_steps', '-')} | "
            f"Batch={ppo.get('batch_size', '-')} | "
            f"Epochs={ppo.get('n_epochs', '-')} | "
            f"γ={ppo.get('gamma', 0):.3f} | "
            f"λ={ppo.get('gae_lambda', 0):.3f} | "
            f"Best={best_params.get('value', 0):.4f}"
        )
    else:
        st.caption("**PPO:** No optimized params - run Hyperparameter Tuning first")

    # Feature summary
    feature_config = st.session_state.get('feature_config', {})
    selected_features = feature_config.get('selected_features', [])
    include_raw = feature_config.get('include_raw_prices', True)
    normalize = feature_config.get('normalize_features', True)

    # Check for prediction features
    prediction_features = [f for f in selected_features if 'predict' in f.lower() or 'lstm' in f.lower() or 'forecast' in f.lower()]
    has_predictions = len(prediction_features) > 0

    feat_summary = f"**Features:** {len(selected_features)} selected"
    if include_raw:
        feat_summary += " + OHLCV"
    if normalize:
        feat_summary += " (normalized)"
    if has_predictions:
        feat_summary += f" | Predictions: {', '.join(prediction_features)}"
    st.caption(feat_summary)

    # Training options - compact row
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        use_saved_model = st.checkbox("Load Saved Model", value=False, key="training_load_saved")
    with c2:
        use_manual_params = st.checkbox("Use Manual Parameters", value=False, key="training_manual",
                                        help="Override Optuna params with manual settings")

    # Main training action
    if use_saved_model:
        try:
            saved_models = [f for f in os.listdir("saved_models") if f.endswith('.zip')]
        except FileNotFoundError:
            saved_models = []

        if saved_models:
            c1, c2 = st.columns([3, 1])
            with c1:
                selected_model = st.selectbox("Model", saved_models, key="training_select_model")
            with c2:
                if st.button("Load", key="training_load_btn"):
                    model_path = os.path.join("saved_models", selected_model)
                    st.session_state.model.load(model_path)
                    st.success(f"Loaded: {selected_model}")
        else:
            st.warning("No saved models in 'saved_models/' directory")

    elif use_manual_params:
        # Manual parameter inputs - compact
        st.caption("Manual PPO Parameters")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            lr = st.number_input("LR", value=3e-4, format="%.1e", key="manual_lr")
        with c2:
            n_steps = st.number_input("Steps", value=1024, step=64, key="manual_steps")
        with c3:
            batch = st.number_input("Batch", value=64, step=32, key="manual_batch")
        with c4:
            epochs = st.number_input("Epochs", value=10, step=1, key="manual_epochs")
        with c5:
            gamma = st.number_input("Gamma", value=0.99, format="%.3f", key="manual_gamma")
        with c6:
            gae = st.number_input("GAE", value=0.95, format="%.2f", key="manual_gae")

        manual_ppo = {
            'learning_rate': lr, 'n_steps': int(n_steps), 'batch_size': int(batch),
            'n_epochs': int(epochs), 'gamma': gamma, 'gae_lambda': gae
        }

        if st.button("Start Training (Manual)", type="primary", key="train_manual_btn"):
            with st.spinner("Training..."):
                run_training(manual_ppo)
            st.success("Training completed")

    else:
        # Default: use Optuna optimized parameters
        if best_params:
            if st.button("Start Training", type="primary", key="train_optuna_btn"):
                with st.spinner("Training with optimized parameters..."):
                    run_training(best_params['params'])
                st.success("Training completed")
        else:
            st.warning("Run Hyperparameter Tuning first to get optimized parameters")

    # Save model option
    with st.expander("Save Model", expanded=False):
        c1, c2 = st.columns([3, 1])
        with c1:
            model_name = st.text_input("Filename", "model_v1.zip", key="training_save_name")
        with c2:
            if st.button("Save", key="training_save_btn"):
                try:
                    os.makedirs("saved_models", exist_ok=True)
                    save_path = os.path.join("saved_models", model_name)
                    st.session_state.model.save(save_path)
                    st.success(f"Saved: {save_path}")
                except Exception as e:
                    st.error(f"Save failed: {e}")
