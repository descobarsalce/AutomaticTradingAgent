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
    display_training_metrics
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




def run_training(ppo_params: Dict[str, Any]) -> None:
    """
    Executes the training process using the training functions module
    """
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    metrics = execute_training(ppo_params, progress_bar, status_placeholder)

    if metrics:
        st.subheader("Parameters Used for Training")
        col1, col2, col3 = st.columns(3)
        index_col = 0
        all_cols = [col1, col2, col3]
        for param, value in ppo_params.items():
            with all_cols[index_col % 3]:
                st.metric(param, value)
                index_col += 1

        display_training_metrics(metrics)

    if hasattr(st.session_state.model.env, '_trade_history'):
        TradingVisualizer.display_trade_history(
            st.session_state.model.env._trade_history, "Training History",
            "training_trade")

    st.session_state.ppo_params = ppo_params
    st.success("Training completed and model saved!")



def hyperparameter_tuning() -> None:
    """
    Interface for hyperparameter optimization using Optuna
    """
    from core.hyperparameter_search import (create_parameter_ranges,
                                          run_hyperparameter_optimization,
                                          display_optimization_results)

    stock_names = st.session_state.stock_names
    train_start_date = st.session_state.train_start_date
    train_end_date = st.session_state.train_end_date
    env_params = st.session_state.env_params

    st.header("Hyperparameter Tuning Options")

    with st.expander("Tuning Configuration", expanded=True):
        trials_number = st.number_input("Number of Trials",
                                      min_value=1,
                                      value=20,
                                      step=1)
        pruning_enabled = st.checkbox("Enable Early Trial Pruning", value=True)
        param_ranges = create_parameter_ranges()
        optimization_metric = st.selectbox(
            "Optimization Metric",
            ["sharpe_ratio", "sortino_ratio", "total_return"],
            help="Metric to optimize during hyperparameter search")

    if st.button("Start Hyperparameter Tuning"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            study = run_hyperparameter_optimization(
                stock_names=stock_names,
                train_start_date=train_start_date,
                train_end_date=train_end_date,
                env_params=env_params,
                param_ranges=param_ranges,
                trials_number=trials_number,
                optimization_metric=optimization_metric,
                progress_bar=progress_bar,
                status_text=status_text,
                pruning_enabled=pruning_enabled
            )

            st.success("Hyperparameter tuning completed!")
            display_optimization_results(study)

        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            logger.exception("Hyperparameter optimization error")


def display_testing_interface(ppo_params, use_optuna_params=False):
    """
    Displays the testing interface and visualization options in a scrollable container
    """
    with st.container():
        st.header("Testing Interface")
        test_col1, test_col2 = st.columns(2)
        with test_col1:
            test_start_date = datetime.combine(
                st.date_input("Test Start Date",
                              value=datetime.now() - timedelta(days=365)),
                datetime.min.time())
        with test_col2:
            test_end_date = datetime.combine(
                st.date_input("Test End Date", value=datetime.now()),
                datetime.min.time())
        st.session_state.test_start_date = test_start_date
        st.session_state.test_end_date = test_end_date

        st.markdown("""
            <style>
                .test-results {
                    max-height: 600px;
                    overflow-y: auto;
                    padding: 1rem;
                    border: 1px solid #e0e0e0;
                    border-radius: 4px;
                }
            </style>
        """,
                    unsafe_allow_html=True)

        if st.button("Test Model"):
            if not os.path.exists("trained_model.zip"):
                st.error("No trained model found. Please train a model first.")
            else:
                if use_optuna_params:
                    ppo_params = st.session_state.ppo_params

                test_results = st.session_state.model.test(
                    stock_names=st.session_state.stock_names,
                    start_date=st.session_state.test_start_date,
                    end_date=st.session_state.test_end_date,
                    env_params=st.session_state.env_params)

            # Display test metrics
            if test_results and 'metrics' in test_results:
                test_results_container = st.container()
                with test_results_container:
                    st.subheader("Test Results Analysis")

                    # Display test trade history
                    if 'info_history' in test_results:
                        TradingVisualizer.display_trade_history(
                            test_results['info_history'], "Test History",
                            "test_trade")

                    # Create tabs for different visualization aspects
                    metrics_tab, trades_tab, analysis_tab = st.tabs([
                        "Performance Metrics", "Trade Analysis",
                        "Technical Analysis"
                    ])

                    with metrics_tab:
                        st.subheader("Performance Metrics")

                    # Display parameters used for testing, automatically sorting into columns:
                    st.subheader("Parameters Used for Testing")
                    col1, col2, col3 = st.columns(3)
                    index_col = 0
                    all_cols = [col1, col2, col3]
                    for param, value in ppo_params.items():
                        with all_cols[index_col % 3]:
                            st.metric(param, value)
                            index_col += 1

                    # Now display the metrics:
                    metrics = test_results['metrics']

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sharpe Ratio",
                                  f"{metrics['sharpe_ratio']:.2f}")
                        st.metric("Max Drawdown",
                                  f"{metrics['max_drawdown']:.2%}")
                    with col2:
                        st.metric("Sortino Ratio",
                                  f"{metrics['sortino_ratio']:.2f}")
                        st.metric("Volatility", f"{metrics['volatility']:.2%}")
                    with col3:
                        if 'information_ratio' in metrics:
                            st.metric("Information Ratio",
                                      f"{metrics['information_ratio']:.2f}")
                        # Calculate and display total return
                        if 'portfolio_history' in test_results:
                            total_return = (
                                (test_results['portfolio_history'][-1] -
                                 test_results['portfolio_history'][0]) /
                                test_results['portfolio_history'][0])
                            st.metric("Total Return", f"{total_return:.2%}")

                    # Display performance charts
                    if 'combined_plot' in test_results:
                        st.plotly_chart(test_results['combined_plot'])
                    with trades_tab:
                        st.subheader("Trading Activity")

                        # Display discrete actions plot
                        if 'action_plot' in test_results:
                            st.plotly_chart(test_results['action_plot'],
                                            use_container_width=True)

                        # Display combined price and actions
                        if 'combined_plot' in test_results:
                            st.plotly_chart(test_results['combined_plot'],
                                            use_container_width=True)

                    with analysis_tab:
                        st.subheader("Technical Analysis")
                        if 'info_history' in test_results:
                            visualizer = TradingVisualizer()

                            # Show correlation analysis if multiple stocks
                            portfolio_data = st.session_state.model.data_handler.fetch_data(
                                st.session_state.stock_names,
                                st.session_state.test_start_date,
                                st.session_state.test_end_date)
                            if len(st.session_state.stock_names) > 1:
                                corr_fig = visualizer.plot_correlation_heatmap(
                                    portfolio_data)
                                st.plotly_chart(corr_fig,
                                                use_container_width=True)

                            # Show drawdown analysis
                            for symbol in st.session_state.stock_names:
                                drawdown_fig = visualizer.plot_performance_and_drawdown(
                                    portfolio_data, symbol)
                                st.plotly_chart(drawdown_fig,
                                                use_container_width=True)