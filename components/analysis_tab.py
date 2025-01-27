import optuna
import streamlit as st
from typing import Dict, Any, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
import os
import traceback

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def hyperparameter_tuning(stock_name: str, train_start_date: datetime,
                         train_end_date: datetime, env_params: Dict[str, Any]) -> None:
    st.header("Hyperparameter Tuning Options")

    with st.expander("Tuning Configuration", expanded=True):
        trials_number = st.number_input("Number of Trials",
                                      min_value=1,
                                      value=20,
                                      step=1)
        pruning_enabled = st.checkbox("Enable Early Trial Pruning", value=True)

        st.subheader("Parameter Search Ranges")

        col1, col2 = st.columns(2)
        with col1:
            lr_min = st.number_input("Learning Rate Min",
                                   value=1e-5,
                                   format="%.1e")
            lr_max = st.number_input("Learning Rate Max",
                                   value=5e-4,
                                   format="%.1e")
            steps_min = st.number_input("Steps Min", value=512, step=64)
            steps_max = st.number_input("Steps Max", value=2048, step=64)
            batch_min = st.number_input("Batch Size Min", value=64, step=32)
            batch_max = st.number_input("Batch Size Max", value=512, step=32)

        with col2:
            epochs_min = st.number_input("Training Epochs Min",
                                       value=3,
                                       step=1)
            epochs_max = st.number_input("Training Epochs Max",
                                       value=10,
                                       step=1)
            gamma_min = st.number_input("Gamma Min",
                                      value=0.90,
                                      step=0.01,
                                      format="%.3f")
            gamma_max = st.number_input("Gamma Max",
                                      value=0.999,
                                      step=0.001,
                                      format="%.3f")
            gae_min = st.number_input("GAE Lambda Min",
                                    value=0.90,
                                    step=0.01,
                                    format="%.2f")
            gae_max = st.number_input("GAE Lambda Max",
                                    value=0.99,
                                    step=0.01,
                                    format="%.2f")

        optimization_metric = st.selectbox(
            "Optimization Metric",
            ["sharpe_ratio", "sortino_ratio", "total_return"],
            help="Metric to optimize during hyperparameter search")

    if st.button("Start Hyperparameter Tuning"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create study with pruning
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner() if pruning_enabled else None)

        def objective(trial: optuna.Trial) -> float:
            try:
                ppo_params = {
                    'learning_rate':
                    trial.suggest_loguniform('learning_rate', lr_min, lr_max),
                    'n_steps':
                    trial.suggest_int('n_steps', steps_min, steps_max),
                    'batch_size':
                    trial.suggest_int('batch_size', batch_min, batch_max),
                    'n_epochs':
                    trial.suggest_int('n_epochs', epochs_min, epochs_max),
                    'gamma':
                    trial.suggest_uniform('gamma', gamma_min, gamma_max),
                    'gae_lambda':
                    trial.suggest_uniform('gae_lambda', gae_min, gae_max),
                }

                status_text.text(
                    f"Trial {trial.number + 1}/{trials_number}: Testing parameters {ppo_params}"
                )

                # Train with current parameters
                metrics = st.session_state.model.train(
                    stock_name=stock_name,
                    start_date=train_start_date,
                    end_date=train_end_date,
                    env_params=env_params,
                    ppo_params=ppo_params)

                # Use Sharpe ratio as optimization metric
                trial_value = metrics.get('sharpe_ratio', float('-inf'))
                progress = (trial.number + 1) / trials_number
                progress_bar.progress(progress)

                return trial_value

            except Exception as e:
                st.error(f"Error in trial {trial.number}: {str(e)}")
                return float('-inf')

        try:
            study.optimize(objective, n_trials=trials_number)

            st.success("Hyperparameter tuning completed!")

            # Create detailed results dataframe
            trials_df = pd.DataFrame([{
                'Trial': t.number,
                'Value': t.value,
                **t.params
            } for t in study.trials if t.value is not None])

            # Display results in tabs
            tab1, tab2, tab3 = st.tabs([
                "Best Parameters", "Optimization History",
                "Parameter Importance"
            ])

            with tab1:
                st.subheader("Best Configuration Found")
                for param, value in study.best_params.items():
                    if param == 'learning_rate':
                        st.metric(f"Best {param}", f"{value:.2e}")
                    elif param in ['gamma', 'gae_lambda']:
                        st.metric(f"Best {param}", f"{value:.4f}")
                    else:
                        st.metric(f"Best {param}", f"{int(value)}")
                st.metric(f"Best {optimization_metric}",
                          f"{study.best_value:.6f}")

            with tab2:
                st.subheader("Trial History")
                history_fig = go.Figure()
                history_fig.add_trace(
                    go.Scatter(x=trials_df.index,
                               y=trials_df['Value'],
                               mode='lines+markers',
                               name='Trial Value'))
                history_fig.update_layout(
                    title='Optimization History',
                    xaxis_title='Trial Number',
                    yaxis_title=optimization_metric.replace('_', ' ').title())
                st.plotly_chart(history_fig)

            with tab3:
                st.subheader("Parameter Importance")
                importance_dict = optuna.importance.get_param_importances(
                    study)
                importance_df = pd.DataFrame({
                    'Parameter':
                    list(importance_dict.keys()),
                    'Importance':
                    list(importance_dict.values())
                }).sort_values('Importance', ascending=True)

                importance_fig = go.Figure()
                importance_fig.add_trace(
                    go.Bar(x=importance_df['Importance'],
                           y=importance_df['Parameter'],
                           orientation='h'))
                importance_fig.update_layout(
                    title='Parameter Importance Analysis',
                    xaxis_title='Relative Importance',
                    yaxis_title='Parameter',
                    height=400)
                st.plotly_chart(importance_fig)

            # Save full results to CSV
            trials_df.to_csv('hyperparameter_tuning_results.csv', index=False)

            # Download button for results
            st.download_button("Download Complete Results CSV",
                               trials_df.to_csv(index=False),
                               "hyperparameter_tuning_results.csv", "text/csv")

            # Save best parameters
            st.session_state.ppo_params = study.best_params

        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            logger.exception("Hyperparameter optimization error")


"""
Technical Analysis Dashboard Component
Handles the visualization and analysis of stock data
"""
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd
import os
from utils.stock_utils import parse_stock_list

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'analysis_error' not in st.session_state:
        st.session_state.analysis_error = None

def safe_data_access(func):
    """Decorator for safe data access with error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"An error occurred: {str(e)}")
            return None
    return wrapper

@safe_data_access
def display_analysis_tab(model):
    """
    Renders the technical analysis dashboard tab
    Args:
        model: The trading model instance
    """
    try:
        initialize_session_state()

        if model is None:
            st.warning("Please initialize the model first.")
            return

        st.session_state.model = model

        st.header("Technical Analysis Dashboard")

        # Add loading indicator
        with st.spinner("Loading analysis components..."):
            # Stock selection with error handling
            viz_stock_input = st.text_input(
                "Stocks to Visualize (comma-separated)",
                value="AAPL, MSFT, GOOGL"
            )

            if not viz_stock_input:
                st.warning("Please enter at least one stock symbol")
                return

            try:
                viz_stocks = parse_stock_list(viz_stock_input)
            except Exception as e:
                st.error(f"Error parsing stock symbols: {str(e)}")
                return

            # Date selection
            try:
                viz_col1, viz_col2 = st.columns(2)
                with viz_col1:
                    viz_start_date = datetime.combine(
                        st.date_input("Analysis Start Date",
                                    value=datetime.now() - timedelta(days=365)),
                        datetime.min.time())
                with viz_col2:
                    viz_end_date = datetime.combine(
                        st.date_input("Analysis End Date", 
                                    value=datetime.now()),
                        datetime.min.time())
            except Exception as e:
                st.error(f"Error setting dates: {str(e)}")
                return

            # Plot controls
            st.subheader("Visualization Options")
            try:
                plot_col1, plot_col2, plot_col3 = st.columns(3)

                with plot_col1:
                    show_rsi = st.checkbox("Show RSI", value=True)
                    show_sma20 = st.checkbox("Show SMA 20", value=True)

                with plot_col2:
                    show_sma50 = st.checkbox("Show SMA 50", value=True)
                    rsi_period = st.slider("RSI Period",
                                        min_value=7,
                                        max_value=21,
                                        value=14) if show_rsi else 14

                with plot_col3:
                    num_columns = st.selectbox("Number of Columns",
                                            options=[1, 2, 3, 4],
                                            index=1)
            except Exception as e:
                st.error(f"Error setting visualization options: {str(e)}")
                return

            if st.button("Generate Analysis"):
                with st.spinner("Generating analysis..."):
                    try:
                        if model.data_handler is None:
                            st.error("Data handler not initialized")
                            return

                        generate_analysis(
                            viz_stocks, viz_start_date, viz_end_date,
                            model, show_rsi, show_sma20, show_sma50,
                            rsi_period, num_columns
                        )
                    except Exception as e:
                        logger.error(f"Analysis generation error: {str(e)}")
                        logger.error(traceback.format_exc())
                        st.error(f"Failed to generate analysis: {str(e)}")

    except Exception as e:
        logger.error(f"Display analysis tab error: {str(e)}")
        logger.error(traceback.format_exc())
        st.error("An error occurred while displaying the analysis tab. Please check the logs for details.")

@safe_data_access
def generate_analysis(viz_stocks, viz_start_date, viz_end_date, model,
                    show_rsi, show_sma20, show_sma50, rsi_period,
                    num_columns):
    """
    Generates and displays technical analysis charts
    """
    st.subheader("Analysis Results")

    analysis_container = st.container()
    with analysis_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create dictionaries to store different types of charts
        price_charts = {}
        volume_charts = {}
        rsi_charts = {}
        ma_charts = {}

        total_stocks = len(viz_stocks)

        # First collect all data and create charts
        for idx, stock in enumerate(viz_stocks):
            status_text.text(f"Processing {stock} ({idx + 1}/{total_stocks})")
            progress_bar.progress((idx + 1) / total_stocks)

            try:
                portfolio_data = model.data_handler.fetch_data(
                    stock, viz_start_date, viz_end_date)

                if not portfolio_data:
                    st.warning(f"No data available for {stock}")
                    continue

                data = model.data_handler.prepare_data()

                if stock not in data:
                    st.warning(f"No prepared data available for {stock}")
                    continue

                stock_data = data[stock]

                # Create charts with error handling
                try:
                    price_charts[stock] = create_price_chart(stock_data, stock)
                    volume_charts[stock] = create_volume_chart(stock_data, stock)
                    if show_rsi:
                        rsi_charts[stock] = create_rsi_chart(stock_data, stock, rsi_period)
                    if show_sma20 or show_sma50:
                        ma_charts[stock] = create_ma_chart(stock_data, stock, show_sma20, show_sma50)
                except Exception as e:
                    logger.error(f"Error creating charts for {stock}: {str(e)}")
                    st.warning(f"Could not create some charts for {stock}")

            except Exception as e:
                logger.error(f"Error processing {stock}: {str(e)}")
                st.warning(f"Error processing {stock}: {str(e)}")

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Display charts using the dynamic grid system
        if any([price_charts, volume_charts, rsi_charts, ma_charts]):
            display_charts_grid(price_charts, "Price Analysis", num_columns)
            display_charts_grid(volume_charts, "Volume Analysis", num_columns)
            display_charts_grid(rsi_charts, "RSI Analysis", num_columns)
            display_charts_grid(ma_charts, "Moving Averages Analysis", num_columns)
        else:
            st.error("No charts could be generated. Please check the data and try again.")

def create_price_chart(data, stock):
    return go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=stock
        )
    ]).update_layout(title=f'{stock} Price History')

def create_volume_chart(data, stock):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name=f'{stock} Volume'))
    fig.update_layout(title=f'{stock} Trading Volume')
    return fig

def create_rsi_chart(data, stock, period):
    if 'RSI' not in data.columns:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=data.index, y=data['RSI'] * 100, name=f'{stock} RSI'))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(title=f'{stock} RSI ({period} periods)')
    return fig

def create_ma_chart(data, stock, show_sma20, show_sma50):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name=f'{stock} Price'))
    if show_sma20 and 'SMA_20' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_20'], name=f'{stock} SMA 20'))
    if show_sma50 and 'SMA_50' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_50'], name=f'{stock} SMA 50'))
    fig.update_layout(title=f'{stock} Moving Averages')
    return fig

def display_charts_grid(charts: Dict[str, go.Figure], title: str,
                       num_columns: int) -> None:
    """
    Displays charts in a grid layout
    """
    if not charts:
        return

    st.subheader(title)
    stocks = list(charts.keys())

    for i in range(0, len(charts), num_columns):
        cols = st.columns(num_columns)
        for j in range(num_columns):
            if i + j < len(stocks):
                with cols[j]:
                    chart = charts[stocks[i + j]]
                    if chart is not None:
                        st.plotly_chart(chart, use_container_width=True)