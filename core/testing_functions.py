"""
Testing Functions Module
Handles model evaluation and visualization of test results
"""
import streamlit as st
import os
import numpy as np
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

        if model is None:
            st.error("No model initialized. Please train or load a model first.")
            return

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

        if test_end_date <= test_start_date:
            st.error("End date must be after start date")
            return

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
            try:
                model_path = "trained_model.zip"
                if not os.path.exists(model_path):
                    st.error("No trained model found. Please train a model first.")
                    return

                if not stock_names:
                    st.error("No stock symbols provided")
                    return

                st.info("Starting model testing...")

                if use_optuna_params and hasattr(st.session_state, 'ppo_params'):
                    ppo_params = st.session_state.ppo_params

                with st.spinner('Testing model...'):
                    test_results = model.test(
                        stock_names=stock_names,
                        start_date=test_start_date,
                        end_date=test_end_date,
                        env_params=env_params)

                    if test_results is None:
                        st.error("Testing failed - no results returned")
                        return

                st.success("Testing completed successfully!")

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
                        metrics_tab, trades_tab, positions_tab, analysis_tab = st.tabs([
                            "Performance Metrics", "Trade Analysis",
                            "Position History", "Technical Analysis"
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
                                st.plotly_chart(test_results['combined_plot'], key="metrics_combined_plot")

                        with trades_tab:
                            st.subheader("Trading Activity")
                            display_trading_activity(test_results)

                        with positions_tab:
                            st.subheader("Position History")
                            display_position_history(test_results, stock_names)

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
                                                    use_container_width=True,
                                                    key="correlation_heatmap")

                                # Show drawdown analysis
                                for idx, symbol in enumerate(stock_names):
                                    drawdown_fig = visualizer.plot_performance_and_drawdown(
                                        portfolio_data, symbol)
                                    st.plotly_chart(drawdown_fig,
                                                    use_container_width=True,
                                                    key=f"drawdown_{symbol}_{idx}")

            except Exception as e:
                st.error(f"Error during testing: {str(e)}")
                return

def display_metrics_grid(metrics: Dict[str, float], test_results: Dict[str, Any]):
    """
    Displays metrics in a grid layout
    """
    def safe_format(value):
        if hasattr(value, 'item'):
            return float(value.item())
        return float(value)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sharpe Ratio", f"{safe_format(metrics['sharpe_ratio']):.2f}")
        st.metric("Max Drawdown", f"{safe_format(metrics['max_drawdown']):.2%}")
    with col2:
        st.metric("Sortino Ratio", f"{safe_format(metrics['sortino_ratio']):.2f}")
        st.metric("Volatility", f"{safe_format(metrics['volatility']):.2%}")
    with col3:
        if 'information_ratio' in metrics:
            st.metric("Information Ratio", f"{safe_format(metrics['information_ratio']):.2f}")
        if 'portfolio_history' in test_results:
            total_return = ((test_results['portfolio_history'][-1] -
                         test_results['portfolio_history'][0]) /
                         test_results['portfolio_history'][0])
            st.metric("Total Return", f"{safe_format(total_return):.2%}")

def display_trading_activity(test_results: Dict[str, Any]):
    """
    Displays trading activity plots
    """
    if 'action_plot' in test_results:
        st.plotly_chart(test_results['action_plot'], use_container_width=True, key="trades_action_plot")

    if 'combined_plot' in test_results:
        st.plotly_chart(test_results['combined_plot'], use_container_width=True, key="trades_combined_plot")

def display_position_history(test_results: Dict[str, Any], stock_names: list):
    """
    Display detailed position history showing all position changes
    """
    import pandas as pd

    if 'info_history' not in test_results:
        st.warning("No position history data available")
        return

    info_history = test_results['info_history']

    # Extract position data from info_history
    position_records = []
    prev_positions = None
    prev_balance = None

    for i, info in enumerate(info_history):
        current_positions = info.get('positions', {})
        current_balance = info.get('balance', 0)
        current_value = info.get('portfolio_value', 0)
        date = info.get('date', i)

        # Check if positions or balance changed
        positions_changed = (prev_positions != current_positions) or (prev_balance != current_balance)

        # Always include first and last record, or when positions changed
        if i == 0 or i == len(info_history) - 1 or positions_changed:
            record = {
                'Date': date,
                'Balance': f"${current_balance:,.2f}",
                'Portfolio Value': f"${current_value:,.2f}",
            }

            # Add each stock position
            for stock in stock_names:
                shares = current_positions.get(stock, 0)
                record[f'{stock} Shares'] = shares

            # Calculate allocation percentages
            if current_value > 0:
                cash_pct = (current_balance / current_value) * 100
                record['Cash %'] = f"{cash_pct:.1f}%"

                for stock in stock_names:
                    shares = current_positions.get(stock, 0)
                    if shares > 0 and 'current_data' in info:
                        current_data = info['current_data']
                        price_key = f'Close_{stock}'
                        if price_key in current_data:
                            stock_value = shares * current_data[price_key]
                            stock_pct = (stock_value / current_value) * 100
                            record[f'{stock} %'] = f"{stock_pct:.1f}%"
                        else:
                            record[f'{stock} %'] = "0.0%"
                    else:
                        record[f'{stock} %'] = "0.0%"

            position_records.append(record)

        prev_positions = current_positions.copy()
        prev_balance = current_balance

    # Create DataFrame
    if position_records:
        df = pd.DataFrame(position_records)

        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Position Changes", len(position_records) - 2)  # Exclude first and last
        with col2:
            st.metric("Start Value", position_records[0]['Portfolio Value'])
        with col3:
            st.metric("End Value", position_records[-1]['Portfolio Value'])

        st.markdown("### Position Changes Over Time")
        st.markdown("*This table shows only the days when positions changed*")

        # Display the full table
        st.dataframe(df, use_container_width=True, height=400)

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Position History CSV",
            data=csv,
            file_name="position_history.csv",
            mime="text/csv"
        )

        # Create position allocation over time chart
        st.markdown("### Portfolio Allocation Over Time")
        create_allocation_chart(info_history, stock_names)
    else:
        st.warning("No position changes detected during testing period")

def create_allocation_chart(info_history: list, stock_names: list):
    """
    Create a stacked area chart showing portfolio allocation over time
    """
    import plotly.graph_objects as go

    dates = []
    cash_values = []
    stock_values = {stock: [] for stock in stock_names}

    for info in info_history:
        date = info.get('date')
        balance = info.get('balance', 0)
        positions = info.get('positions', {})
        current_data = info.get('current_data', {})

        dates.append(date)
        cash_values.append(balance)

        for stock in stock_names:
            shares = positions.get(stock, 0)
            price_key = f'Close_{stock}'
            if shares > 0 and price_key in current_data:
                value = shares * current_data[price_key]
            else:
                value = 0
            stock_values[stock].append(value)

    # Create stacked area chart
    fig = go.Figure()

    # Add cash
    fig.add_trace(go.Scatter(
        x=dates,
        y=cash_values,
        mode='lines',
        name='Cash',
        stackgroup='one',
        fillcolor='lightblue'
    ))

    # Add each stock
    colors = ['lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
    for idx, stock in enumerate(stock_names):
        fig.add_trace(go.Scatter(
            x=dates,
            y=stock_values[stock],
            mode='lines',
            name=stock,
            stackgroup='one',
            fillcolor=colors[idx % len(colors)]
        ))

    fig.update_layout(
        title='Portfolio Allocation Over Time',
        xaxis_title='Date',
        yaxis_title='Value ($)',
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True, key="allocation_chart")

def perform_monte_carlo_analysis(model, data, num_simulations=100):
    """
    Performs Monte Carlo analysis by running multiple backtest simulations
    """
    results = []
    for _ in range(num_simulations):
        # Randomize initial conditions slightly
        modified_data = data.copy()
        modified_data *= (1 + np.random.normal(0, 0.001, size=data.shape))

        # Run backtest with modified data
        simulation_result = model.test(modified_data)
        results.append(simulation_result)

    return results

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