
"""
Training Interface Component
Handles the model training and hyperparameter tuning interface
"""
import streamlit as st
from datetime import datetime, timedelta
import optuna
import pandas as pd
import os
import numpy as np
from typing import Dict, Any, Optional

from utils.callbacks import ProgressBarCallback
from core.visualization import TradingVisualizer
from utils.stock_utils import parse_stock_list
from core.base_agent import UnifiedTradingAgent
from core.training_functions import (initialize_training, execute_training,
                                   get_training_parameters, 
                                   display_training_metrics, run_training)
from core.hyperparameter_search import hyperparameter_tuning
from core.testing_functions import display_testing_interface

def display_training_tab():
    """
    Renders the training interface tab
    """
    st.header("Trading Agent Configuration")

    # Input parameters
    st.subheader("Training Options")
    stock_names = st.text_input("Training Stock Symbol",
                               value="AAPL,MSFT,TSLA,GOOG,NVDA")
    st.session_state.stock_names = parse_stock_list(stock_names)

    # Environment parameters
    st.header("Environment Parameters")
    col1, col2 = st.columns(2)
    with col1:
        initial_balance = st.number_input("Initial Balance", value=10000)

    with col2:
        transaction_cost = st.number_input("Transaction Cost",
                                         value=0.01,
                                         step=0.001)

    st.session_state.env_params = {
        'initial_balance': initial_balance,
        'transaction_cost': transaction_cost,
        'use_position_profit': False,
        'use_holding_bonus': False,
        'use_trading_penalty': False
    }

    # Training period selection
    st.subheader("Training Period")
    train_col1, train_col2 = st.columns(2)
    with train_col1:
        train_start_date = datetime.combine(
            st.date_input("Training Start Date",
                         value=datetime.now() - timedelta(days=365 * 5)),
            datetime.min.time())
    with train_col2:
        train_end_date = datetime.combine(
            st.date_input("Training End Date",
                         value=datetime.now() - timedelta(days=365 + 1)),
            datetime.min.time())

    st.session_state.train_start_date = train_start_date
    st.session_state.train_end_date = train_end_date

    tab1, tab2 = st.tabs(["Manual Parameters", "Hyperparameter Tuning"])

    with tab1:
        st.header("Agent Parameters")
        col1, col2 = st.columns(2)
        with col1:
            use_optuna_params = st.checkbox("Use Optuna Optimized Parameters", value=False)
        with col2:
            use_saved_model = st.checkbox("Load Saved Model", value=False)

        if use_saved_model:
            saved_models = [f for f in os.listdir("saved_models") if f.endswith('.zip')]
            if saved_models:
                selected_model = st.selectbox("Select Model", saved_models)
                if st.button("Load Model"):
                    model_path = os.path.join("saved_models", selected_model)
                    st.session_state.model.load(model_path)
                    st.success(f"Model loaded from {model_path}")
            else:
                st.warning("No saved models found")
        elif not use_optuna_params:
            ppo_params = get_training_parameters(use_optuna_params)
            if st.button("Start Training"):
                run_training(ppo_params)
                
            st.write("")  # Add spacing
            model_name = st.text_input("Save model as", "model_v1.zip")
            if st.button("Save Model", use_container_width=True):
                save_path = os.path.join("saved_models", model_name)
                st.session_state.model.save(save_path)
                st.success(f"Model saved to {save_path}")
        else:
            from core.hyperparameter_search import load_best_params
            best_params = load_best_params()
            if best_params is not None:
                st.session_state.ppo_params = best_params['params']
                if st.button("Start Training"):
                    run_training(st.session_state.ppo_params)
            else:
                st.warning("No optimized parameters found. Please run hyperparameter tuning first.")

    with tab2:
        hyperparameter_tuning()

    if st.session_state.ppo_params is not None:
        display_testing_interface(st.session_state.model,
                                st.session_state.stock_names,
                                st.session_state.env_params,
                                st.session_state.ppo_params,
                                use_optuna_params=use_optuna_params)

    # Display code execution interface
    from components.execution_window_ui import display_execution_window
    display_execution_window()
