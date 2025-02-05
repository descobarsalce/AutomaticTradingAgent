
"""
Testing Functions Module
Handles model evaluation and visualization of test results
"""
import streamlit as st
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import pandas as pd

from core.visualization import TradingVisualizer

def display_testing_interface(model, stock_names, env_params, ppo_params, use_optuna_params=False):
    """
    Displays the testing interface and handles test execution
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

        # Store dates in session state
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
        """, unsafe_allow_html=True)

        if st.button("Test Model"):
            if not os.path.exists("trained_model.zip"):
                st.error("No trained model found. Please train a model first.")
            else:
                if use_optuna_params:
                    ppo_params = st.session_state.ppo_params

                test_results = model.test(
                    stock_names=stock_names,
                    start_date=test_start_date,
                    end_date=test_end_date,
                    env_params=env_params)

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

                            # Display parameters used for testing
                            st.subheader("Parameters Used for Testing")
                            for param, value in ppo_params.items():
                                st.metric(param, value)

                            # Display metrics
                            metrics = test_results['metrics']
                            display_metrics_grid(metrics, test_results)

                            # Display performance charts
                            if 'combined_plot' in test_results:
                                st.plotly_chart(test_results['combined_plot'])

                        with trades_tab:
                            st.subheader("Trading Activity")
                            display_trading_activity(test_results)

                        with analysis_tab:
                            st.subheader("Technical Analysis")
                            if 'info_history' in test_results:
                                visualizer = TradingVisualizer()

                                # Show correlation analysis if multiple stocks
                                portfolio_data = st.session_state.data_handler.fetch_data(
                                    stock_names,
                                    test_start_date,
                                    test_end_date)
                                if len(stock_names) > 1:
                                    corr_fig = visualizer.plot_correlation_heatmap(
                                        portfolio_data)
                                    st.plotly_chart(corr_fig,
                                                    use_container_width=True)

                                # Show drawdown analysis
                                for symbol in stock_names:
                                    drawdown_fig = visualizer.plot_performance_and_drawdown(
                                        portfolio_data, symbol)
                                    st.plotly_chart(drawdown_fig,
                                                    use_container_width=True)

def display_metrics_grid(metrics: Dict[str, float], test_results: Dict[str, Any]):
    """
    Displays metrics in a grid layout
    """
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    with col2:
        st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
        st.metric("Volatility", f"{metrics['volatility']:.2%}")
    with col3:
        if 'information_ratio' in metrics:
            st.metric("Information Ratio", f"{metrics['information_ratio']:.2f}")
        if 'portfolio_history' in test_results:
            total_return = ((test_results['portfolio_history'][-1] -
                         test_results['portfolio_history'][0]) /
                         test_results['portfolio_history'][0])
            st.metric("Total Return", f"{total_return:.2%}")

def display_trading_activity(test_results: Dict[str, Any]):
    """
    Displays trading activity plots
    """
    if 'action_plot' in test_results:
        st.plotly_chart(test_results['action_plot'],
                       use_container_width=True)

    if 'combined_plot' in test_results:
        st.plotly_chart(test_results['combined_plot'],
                       use_container_width=True)
def perform_walk_forward_analysis(model, data, window_size=252, step_size=21):
    """
    Performs walk-forward analysis using sliding windows.
    Args:
        window_size: Training window size (default 1 year)
        step_size: Forward testing period (default 1 month)
    """
    results = []
    total_periods = len(data) - window_size
    
    for i in range(0, total_periods, step_size):
        # Training period
        train_start = i
        train_end = i + window_size
        
        # Testing period
        test_start = train_end
        test_end = min(test_start + step_size, len(data))
        
        # Train on window
        train_data = data.iloc[train_start:train_end]
        model.train(train_data)
        
        # Test on unseen data
        test_data = data.iloc[test_start:test_end]
        period_results = model.test(test_data)
        results.append(period_results)
        
    return results
