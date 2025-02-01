"""
Training Interface Component
Handles the model training and hyperparameter tuning interface
"""
import logging
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import os
import traceback

from utils.callbacks import ProgressBarCallback
from core.visualization import TradingVisualizer
from core.base_agent import UnifiedTradingAgent
from core.orchestrator import TradingSystemOrchestrator
from utils.stock_utils import parse_stock_list

logger = logging.getLogger(__name__)

def initialize_orchestrator(stock_names: list, train_start_date: datetime, 
                          train_end_date: datetime, env_params: Dict[str, Any]) -> Optional[TradingSystemOrchestrator]:
    """Initialize the trading system orchestrator with error handling."""
    try:
        return TradingSystemOrchestrator(
            lambda: UnifiedTradingAgent().create_env(
                stock_names=stock_names,
                start_date=train_start_date,
                end_date=train_end_date,
                env_params=env_params
            )
        )
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Failed to initialize trading system: {str(e)}")
        return None

def display_training_tab():
    """
    Renders the training interface tab
    """
    try:
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

        # Initialize orchestrator if not exists or if parameters changed
        if ('orchestrator' not in st.session_state or 
            st.session_state.get('last_stock_names') != st.session_state.stock_names or
            st.session_state.get('last_train_start') != train_start_date or
            st.session_state.get('last_train_end') != train_end_date):

            st.session_state.orchestrator = initialize_orchestrator(
                st.session_state.stock_names,
                train_start_date,
                train_end_date,
                st.session_state.env_params
            )

            # Update last known parameters
            st.session_state.last_stock_names = st.session_state.stock_names
            st.session_state.last_train_start = train_start_date
            st.session_state.last_train_end = train_end_date

        if st.session_state.orchestrator is None:
            st.error("Trading system not initialized. Please check the logs and try again.")
            return

        with tab1:
            st.header("Agent Parameters")
            use_optuna_params = st.checkbox("Use Optuna Optimized Parameters",
                                         value=False)

            if st.button("Start Training"):
                progress_bar = st.progress(0)
                status_placeholder = st.empty()

                try:
                    # Create progress callback
                    progress_callback = ProgressBarCallback(
                        total_timesteps=(train_start_date - train_end_date).days,
                        progress_bar=progress_bar,
                        status_placeholder=status_placeholder
                    )

                    # Run training through orchestrator
                    results = st.session_state.orchestrator.run_training_pipeline(
                        hyperparameters=st.session_state.ppo_params if use_optuna_params else None
                    )

                    if results["status"] == "success":
                        display_training_metrics(results["metrics"])
                        st.success("Training completed successfully!")
                    else:
                        st.error(f"Training failed: {results.get('error', 'Unknown error')}")

                except Exception as e:
                    logger.error(f"Training error: {str(e)}\n{traceback.format_exc()}")
                    st.error(f"Training failed with error: {str(e)}")

        with tab2:
            st.header("Hyperparameter Tuning Options")

            with st.expander("Tuning Configuration", expanded=True):
                trials_number = st.number_input("Number of Trials",
                                            min_value=1,
                                            value=20,
                                            step=1)

                optimization_metric = st.selectbox(
                    "Optimization Metric",
                    ["sharpe_ratio", "sortino_ratio", "total_return"],
                    help="Metric to optimize during hyperparameter search"
                )

            if st.button("Start Hyperparameter Tuning"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Configure optimization
                    config = {
                        "n_trials": trials_number,
                        "metric": optimization_metric,
                    }

                    # Run optimization through orchestrator
                    results = st.session_state.orchestrator.run_optimization_pipeline(config)

                    if "best_params" in results:
                        st.session_state.ppo_params = results["best_params"]
                        display_optimization_results(results)
                        st.success("Hyperparameter optimization completed!")
                    else:
                        st.error("Optimization failed to find best parameters")

                except Exception as e:
                    logger.error(f"Optimization error: {str(e)}\n{traceback.format_exc()}")
                    st.error(f"Optimization failed with error: {str(e)}")

        if hasattr(st.session_state, 'model'):
            display_testing_interface()

    except Exception as e:
        logger.error(f"Error in training tab: {str(e)}\n{traceback.format_exc()}")
        st.error(f"An error occurred: {str(e)}")

def display_training_metrics(metrics: Dict[str, float]) -> None:
    """
    Displays the training metrics in a formatted layout
    """
    try:
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            st.metric("Maximum Drawdown", f"{metrics['max_drawdown']:.2%}")
        with metrics_col2:
            st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
            st.metric("Volatility", f"{metrics['volatility']:.2%}")
        with metrics_col3:
            st.metric("Total Return", f"{metrics['total_return']:.2%}")
            st.metric("Final Portfolio Value", f"${metrics['final_value']:,.2f}")
    except Exception as e:
        logger.error(f"Error displaying metrics: {str(e)}")
        st.error("Failed to display metrics")

def display_optimization_results(results: Dict[str, Any]) -> None:
    """Display optimization results in a structured format"""
    try:
        tab1, tab2 = st.tabs(["Best Parameters", "Optimization History"])

        with tab1:
            st.subheader("Best Configuration Found")
            col1, col2 = st.columns(2)

            for param, value in results["best_params"].items():
                with col1:
                    if param == "learning_rate":
                        st.metric(f"Best {param}", f"{value:.2e}")
                    else:
                        st.metric(f"Best {param}", f"{value}")

            st.metric(f"Best {results['optimization_metric']}", 
                    f"{results['best_value']:.6f}")

        with tab2:
            if "trials_history" in results:
                history_df = pd.DataFrame(results["trials_history"])

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=history_df.index,
                        y=history_df["value"],
                        mode="lines+markers",
                        name="Trial Value"
                    )
                )

                fig.update_layout(
                    title="Optimization History",
                    xaxis_title="Trial Number",
                    yaxis_title=results["optimization_metric"].replace("_", " ").title()
                )

                st.plotly_chart(fig)
    except Exception as e:
        logger.error(f"Error displaying optimization results: {str(e)}")
        st.error("Failed to display optimization results")

def display_testing_interface():
    """
    Displays the testing interface and visualization options
    """
    try:
        st.header("Testing Interface")
        test_col1, test_col2 = st.columns(2)

        with test_col1:
            test_start_date = datetime.combine(
                st.date_input("Test Start Date",
                            value=datetime.now() - timedelta(days=365)),
                datetime.min.time())
        with test_col2:
            test_end_date = datetime.combine(
                st.date_input("Test End Date", 
                            value=datetime.now()),
                datetime.min.time())

        if st.button("Test Model"):
            try:
                results = st.session_state.orchestrator.run_testing_pipeline(
                    os.path.join(os.getcwd(), "trained_model.zip")
                )

                if "evaluation_results" in results:
                    display_test_results(results["evaluation_results"])
                    display_benchmark_comparison(results["benchmark_comparison"])
                else:
                    st.error("Testing failed to produce results")

            except Exception as e:
                logger.error(f"Model testing error: {str(e)}\n{traceback.format_exc()}")
                st.error(f"Testing failed with error: {str(e)}")

    except Exception as e:
        logger.error(f"Error in testing interface: {str(e)}")
        st.error("Failed to display testing interface")

def display_test_results(results: Dict[str, Any]) -> None:
    """Display test evaluation results"""
    try:
        st.subheader("Test Results")

        # Display metrics
        col1, col2, col3 = st.columns(3)
        metrics = results["metrics"]

        with col1:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            st.metric("Maximum Drawdown", f"{metrics['max_drawdown']:.2%}")
        with col2:
            st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
            st.metric("Volatility", f"{metrics['volatility']:.2%}")
        with col3:
            st.metric("Total Return", f"{metrics['total_return']:.2%}")
            if "win_rate" in metrics:
                st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
    except Exception as e:
        logger.error(f"Error displaying test results: {str(e)}")
        st.error("Failed to display test results")

def display_benchmark_comparison(comparison: Dict[str, Any]) -> None:
    """Display benchmark comparison results"""
    try:
        st.subheader("Benchmark Comparison")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Relative Sharpe", 
                    f"{comparison['relative_sharpe']:.2f}")
        with col2:
            st.metric("Relative Return", 
                    f"{comparison['relative_return']:.2%}")
    except Exception as e:
        logger.error(f"Error displaying benchmark comparison: {str(e)}")
        st.error("Failed to display benchmark comparison")