"""
Trading Agent Web Interface
A Streamlit-based dashboard for configuring and managing reinforcement learning trading agents.

System Architecture:
- Web Interface Layer (Streamlit): Provides UI for configuration and visualization
- Agent Layer (UnifiedTradingAgent): Implements trading strategy and training logic
- Environment Layer (SimpleTradingEnv): Simulates market interactions
- Data Layer: Handles market data acquisition and preprocessing

Key Components:
1. Configuration Interface:
   - Agent hyperparameter tuning
   - Environment settings adjustment
   - Training period selection

2. Training Pipeline:
   - Data preparation
   - Model training with progress tracking
   - Performance metrics calculation

3. Testing Interface:
   - Model evaluation on unseen data
   - Performance visualization
   - Trading behavior analysis

4. Monitoring System:
   - Real-time training progress
   - Portfolio value tracking
   - Transaction logging

Usage Example:
```python
# Initialize the application
streamlit run main.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=true

# Configure agent parameters in UI:
# - Set stock symbol (e.g., 'AAPL')
# - Adjust initial balance
# - Tune learning parameters
# - Select date ranges

# Train the agent:
# Click "Start Training" to begin the training process
# Monitor progress in the sidebar logs

# Test the agent:
# Click "Test Model" to evaluate on new data
# View performance metrics and visualizations
```
"""

import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any
from utils.callbacks import ProgressBarCallback
from core.base_agent import UnifiedTradingAgent

import optuna
from sqlalchemy import func, distinct
from models.database import Session
from models.models import StockData
from utils.stock_utils import parse_stock_list

# Configure logging
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def init_session_state() -> None:
    """
    Initialize Streamlit session state variables for persistent storage across reruns.

    Initializes:
        - log_messages: List[str] - Chronological log messages
        - ppo_params: Dict[str, Union[float, int, bool]] - PPO algorithm configuration
        - model: UnifiedTradingAgent - Trading agent model instance

    Implementation:
        The function checks for each required key in st.session_state and
        initializes it if missing. This ensures persistence across Streamlit reruns
        while avoiding reinitializing existing state.

    Example:
        ```python
        # Initialize state at app startup
        init_session_state()

        # Access state variables
        model = st.session_state.model
        logs = st.session_state.log_messages
        ```
    """
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    if 'ppo_params' not in st.session_state:
        st.session_state.ppo_params = None
    if 'model' not in st.session_state:
        st.session_state.model = UnifiedTradingAgent()


class StreamlitLogHandler(logging.Handler):
    """
    Custom logging handler that redirects log messages to Streamlit's interface.

    Maintains a fixed-size buffer of recent log messages in the session state.
    Implements the observer pattern to capture logs from all components.

    Attributes:
        MAX_LOGS: int - Maximum number of logs to maintain in memory (100)
        format: Callable - Log message formatter function

    Implementation Details:
        - Uses Streamlit session state for persistence
        - Implements circular buffer behavior for log storage
        - Thread-safe for concurrent logging

    Example:
        ```python
        # Setup logging
        handler = StreamlitLogHandler()
        logger.addHandler(handler)

        # Log messages will appear in Streamlit sidebar
        logger.info("Training started...")
        ```
    """

    MAX_LOGS: int = 100

    def emit(self, record: logging.LogRecord) -> None:
        """
        Process a log record by formatting it and adding to the session state.

        Args:
            record (logging.LogRecord): The log record containing:
                - msg: str - The log message
                - levelno: int - Logging level number
                - created: float - Time when the log was created
                - args: tuple - Message format arguments

        Implementation:
            1. Formats the log record using the handler's formatter
            2. Appends to session state log buffer
            3. Maintains maximum log count by removing oldest entries
            4. Handles formatting exceptions gracefully

        Raises:
            Exception: If logging fails, error is printed but not propagated
        """
        try:
            log_entry: str = self.format(record)
            if 'log_messages' in st.session_state:
                st.session_state.log_messages.append(log_entry)
                if len(st.session_state.log_messages) > self.MAX_LOGS:
                    st.session_state.log_messages = st.session_state.log_messages[
                        -self.MAX_LOGS:]
            print(log_entry)  # Backup output
        except Exception as e:
            print(f"Logging error: {e}")


def hyperparameter_tuning(stock_name: str, train_start_date: datetime,
                          train_end_date: datetime,
                          env_params: Dict[str, Any]) -> None:
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
        status_text = st.empty()

        study = optuna.create_study(direction='maximize')

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


def display_database_explorer():
    """Display the database explorer interface"""
    st.title("Database Explorer")
    st.header("Database Statistics")

    # Initialize database session
    session = Session()

    col1, col2, col3 = st.columns(3)

    # Total unique symbols
    unique_symbols = session.query(func.count(distinct(
        StockData.symbol))).scalar()
    col1.metric("Total Unique Symbols", unique_symbols)

    # Date range
    min_date = session.query(func.min(StockData.date)).scalar()
    max_date = session.query(func.max(StockData.date)).scalar()
    if min_date and max_date:
        date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        col2.metric("Date Range", date_range)

    # Database file size
    if os.path.exists('trading_data.db'):
        db_size = os.path.getsize('trading_data.db') / (1024 * 1024
                                                        )  # Convert to MB
        col3.metric("Database Size", f"{db_size:.2f} MB")

    # Stock Data Summary
    st.header("Stock Data Summary")

    # Query for stock summary information
    stock_summary = []
    symbols = [
        row[0] for row in session.query(distinct(StockData.symbol)).all()
    ]

    for symbol in symbols:
        # Get statistics for each stock
        symbol_data = session.query(
            StockData.symbol,
            func.min(StockData.date).label('start_date'),
            func.max(StockData.date).label('end_date'),
            func.count(StockData.id).label('data_points'),
            func.max(StockData.last_updated).label('last_update')).filter(
                StockData.symbol == symbol).group_by(StockData.symbol).first()

        if symbol_data:
            # Calculate coverage percentage
            total_days = (symbol_data.end_date -
                          symbol_data.start_date).days + 1
            coverage = (symbol_data.data_points / total_days) * 100

            stock_summary.append({
                'Symbol':
                symbol,
                'Start Date':
                symbol_data.start_date.strftime('%Y-%m-%d'),
                'End Date':
                symbol_data.end_date.strftime('%Y-%m-%d'),
                'Data Points':
                symbol_data.data_points,
                'Coverage (%)':
                f"{coverage:.1f}%",
                'Last Update':
                symbol_data.last_update.strftime('%Y-%m-%d %H:%M:%S')
            })

    if stock_summary:
        # Convert to DataFrame and display
        summary_df = pd.DataFrame(stock_summary)
        st.dataframe(summary_df,
                     column_config={
                         'Symbol':
                         st.column_config.TextColumn('Symbol', width='small'),
                         'Coverage (%)':
                         st.column_config.TextColumn('Coverage (%)',
                                                     width='small'),
                         'Data Points':
                         st.column_config.NumberColumn('Data Points',
                                                       format="%d")
                     })

        # Add download button for the summary
        csv = summary_df.to_csv(index=False)
        st.download_button("Download Summary CSV",
                           csv,
                           "stock_data_summary.csv",
                           "text/csv",
                           key='download-csv')
    else:
        st.info("No stock data available in the database.")

    # Query Interface
    st.header("Query Interface")

    # Symbol selection
    symbols = [
        row[0] for row in session.query(distinct(StockData.symbol)).all()
    ]
    if symbols:
        selected_symbol = st.selectbox("Select Symbol", symbols)

        # Date range selection
        date_col1, date_col2 = st.columns(2)
        start_date = date_col1.date_input("Start Date",
                                          min_date if min_date else None)
        end_date = date_col2.date_input("End Date",
                                        max_date if max_date else None)

        if st.button("Query Data"):
            # Fetch data
            query_data = session.query(StockData).filter(
                StockData.symbol == selected_symbol, StockData.date
                >= start_date, StockData.date
                <= end_date).order_by(StockData.date).all()

            # Convert to DataFrame
            df = pd.DataFrame([{
                'Date': record.date,
                'Open': record.open,
                'High': record.high,
                'Low': record.low,
                'Close': record.close,
                'Volume': record.volume
            } for record in query_data])

            if not df.empty:
                # Calculate basic statistics
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                stats_col1.metric("Average Price",
                                  f"${df['Close'].mean():.2f}")
                stats_col2.metric("Highest Price", f"${df['High'].max():.2f}")
                stats_col3.metric("Lowest Price", f"${df['Low'].min():.2f}")

                # Display table view
                st.subheader("Table View")
                st.dataframe(df)

                # Display chart view
                st.subheader("Chart View")
                fig = go.Figure(data=[
                    go.Candlestick(x=df['Date'],
                                   open=df['Open'],
                                   high=df['High'],
                                   low=df['Low'],
                                   close=df['Close'])
                ])

                fig.update_layout(title=f'{selected_symbol} Price History',
                                  yaxis_title='Price ($)',
                                  template='plotly_dark',
                                  height=600)

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for the selected criteria.")


def main() -> None:
    init_session_state()

    st.title("Trading Analysis and Agent Platform")

    # Create tabs for Technical Analysis, Model Training, and Database Explorer
    tab_training, tab_analysis, tab_database = st.tabs(
        ["Model Training", "Technical Analysis", "Database Explorer"])

    with tab_analysis:
        from components.analysis_tab import display_analysis_tab
        display_analysis_tab(st.session_state.model)

    with tab_training:
        from components.training_tab import display_training_tab
        display_training_tab()

    with tab_database:
        display_database_explorer()


if __name__ == "__main__":
    main()
