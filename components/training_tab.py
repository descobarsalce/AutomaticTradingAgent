"""
Training Interface Component
Handles the model training and hyperparameter tuning interface
"""
import streamlit as st
import logging

logger = logging.getLogger(__name__)
from datetime import datetime, timedelta
import optuna
import pandas as pd
import os
import numpy as np
from typing import Dict, Any, Optional
import logging

from utils.callbacks import ProgressBarCallback
from core.visualization import TradingVisualizer
from utils.stock_utils import parse_stock_list
from core.base_agent import UnifiedTradingAgent
from core.training_functions import (initialize_training, execute_training,
                                     get_training_parameters,
                                     display_training_metrics, run_training)
from core.hyperparameter_search import hyperparameter_tuning


def display_training_tab():
    """
    Renders the model training interface tab
    """
    start_time = datetime.now()
    logger.info("Initializing training tab...")
    st.header("Model Training")

    # Add a checkbox for enabling/disabling logging
    enable_logging = st.checkbox("Enable Logging", value=False, key="training_enable_logging")
    st.session_state.enable_logging = enable_logging

    # Set the logging level based on the checkbox
    if enable_logging:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.CRITICAL)

    # Display current configuration from tuning tab
    st.subheader("Current Configuration")
    st.info(
        f"**Stocks:** {', '.join(st.session_state.get('stock_names', ['AAPL', 'MSFT']))}\n\n"
        f"**Initial Balance:** ${st.session_state.get('env_params', {}).get('initial_balance', 10000):,.2f}\n\n"
        f"**Transaction Cost:** {st.session_state.get('env_params', {}).get('transaction_cost', 0.01)}\n\n"
        f"**Training Period:** {st.session_state.get('train_start_date', 'Not set').strftime('%Y-%m-%d') if hasattr(st.session_state.get('train_start_date', 'Not set'), 'strftime') else 'Not set'} to "
        f"{st.session_state.get('train_end_date', 'Not set').strftime('%Y-%m-%d') if hasattr(st.session_state.get('train_end_date', 'Not set'), 'strftime') else 'Not set'}"
    )

    st.caption("💡 To change these parameters, go to the Hyperparameter Tuning tab")

    # Training parameters section
    st.header("Training Mode")
    col1, col2 = st.columns(2)
    with col1:
        use_optuna_params = st.checkbox("Use Optuna Optimized Parameters",
                                        value=False,
                                        key="training_use_optuna_params")
    with col2:
        use_saved_model = st.checkbox("Load Saved Model", value=False, key="training_use_saved_model")

    if use_saved_model:
        saved_models = [
            f for f in os.listdir("saved_models") if f.endswith('.zip')
        ]
        if saved_models:
            selected_model = st.selectbox("Select Model", saved_models, key="training_select_model")
            if st.button("Load Model", key="training_load_model"):
                model_path = os.path.join("saved_models", selected_model)
                st.session_state.model.load(model_path)
                st.success(f"Model loaded from {model_path}")
        else:
            st.warning("No saved models found")
    elif not use_optuna_params:
        ppo_params = get_training_parameters(use_optuna_params)
        if st.button("Start Training", key="training_start_training_manual"):
            run_training(ppo_params)
            st.info(
                "Training completed. Check logs to see if trades were registered."
            )

        st.write("")  # Add spacing
        model_name = st.text_input("Save model as", "model_v1.zip", key="training_model_name")
        if st.button("Save Model", use_container_width=True, key="training_save_model"):
            save_path = os.path.join("saved_models", model_name)
            st.session_state.model.save(save_path)
            st.success(f"Model saved to {save_path}")
    else:
        from core.hyperparameter_search import load_best_params
        best_params = load_best_params()
        if best_params is not None:
            st.session_state.ppo_params = best_params['params']
            if st.button("Start Training", key="training_start_training_optuna"):
                run_training(st.session_state.ppo_params)
        else:
            st.warning(
                "No optimized parameters found. Please run hyperparameter tuning first."
            )
