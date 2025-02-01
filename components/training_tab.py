"""
Training Interface Component
Handles the model training and hyperparameter tuning interface
"""
import streamlit as st
from datetime import datetime, timedelta
import optuna
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt

from utils.callbacks import ProgressBarCallback

from core.visualization import TradingVisualizer

from core.base_agent import UnifiedTradingAgent
import os
import numpy as np
from utils.stock_utils import parse_stock_list

from core.training_functions import (
    initialize_training,
    execute_training,
    get_training_parameters,
    display_training_metrics,
    run_training
)

def display_training_tab():
    """
    Renders the training interface tab
    """
    # Initialize session state with saved parameters if they exist
    if 'ppo_params' not in st.session_state:
        from core.hyperparameter_search import load_best_params
        best_params = load_best_params()
        if best_params:
            st.session_state.ppo_params = best_params['params']

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
        use_optuna_params = st.checkbox("Use Optuna Optimized Parameters",
                                        value=False)
        if not use_optuna_params:
            ppo_params = get_training_parameters(use_optuna_params)
            if st.button("Start Training"):
                run_training(ppo_params)
        else:
            if st.button("Start Training"):
                if st.session_state.ppo_params is None:
                    st.warning(
                        "Please run hyperparameter tuning before training model."
                    )
                else:
                    # Note that this will only work in the optimizaiton has already been run so that it has
                    run_training(st.session_state.ppo_params)

    with tab2:
        hyperparameter_tuning()

    if st.session_state.ppo_params is not None:
        display_testing_interface(st.session_state.ppo_params,
                                          use_optuna_params)

    # Add Python code execution interface
    st.header("Data Analysis Console")
    with st.expander("Python Code Execution", expanded=True):
        code = st.text_area(
            "Enter Python code:",
            height=300,
            help="Access data via st.session_state.model.data_handler")

        # Initialize persistent namespace in session state if not exists
        if 'code_namespace' not in st.session_state:
            st.session_state.code_namespace = {
                'np': np,
                'pd': pd,
                'plt': plt,
                'go': go,
                'vars': {},  # For user-defined variables
            }

        if st.button("Execute Code"):
            try:
                # Update namespace with latest session state
                st.session_state.code_namespace.update({
                    'data_handler':
                    st.session_state.model.data_handler,
                    'stock_names':
                    st.session_state.stock_names,
                    'train_start_date':
                    st.session_state.train_start_date,
                    'train_end_date':
                    st.session_state.train_end_date,
                    'test_start_date':
                    st.session_state.test_start_date,
                    'test_end_date':
                    st.session_state.test_end_date,
                    'env_params':
                    st.session_state.env_params,
                    'model':
                    st.session_state.model,
                    'vars':
                    st.session_state.
                    code_namespace['vars'],  # Preserve user variables
                })

                # Create reference to vars dict for easier access
                locals().update(st.session_state.code_namespace['vars'])

                # Create string buffer to capture print output
                import io
                import sys
                output_buffer = io.StringIO()
                original_stdout = sys.stdout
                sys.stdout = output_buffer

                # Execute the code and capture output
                with st.spinner("Executing code..."):
                    exec(code, globals(), st.session_state.code_namespace)

                    # Save all newly defined variables
                    st.session_state.code_namespace['vars'].update({
                        k: v
                        for k, v in st.session_state.code_namespace.items()
                        if k not in [
                            'np', 'pd', 'plt', 'go', 'data_handler',
                            'stock_names', 'train_start_date',
                            'train_end_date', 'test_start_date',
                            'test_end_date', 'env_params', 'model', 'vars'
                        ]
                    })

                    # Display any generated plots
                    if 'plt' in st.session_state.code_namespace:
                        st.pyplot(plt.gcf())
                        plt.close()

                    # Get and display the captured output
                    sys.stdout = original_stdout
                    output = output_buffer.getvalue()
                    if output:
                        st.text_area("Output:", value=output, height=250)

            except Exception as e:
                st.error(f"Error executing code: {str(e)}")
            finally:
                # Ensure stdout is restored
                sys.stdout = original_stdout











