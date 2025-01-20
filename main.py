"""
Trading Agent Web Interface
A Streamlit-based dashboard for configuring and managing reinforcement learning trading agents.

System Architecture:
- Web Interface Layer (Streamlit): Provides UI for configuration and visualization
- Agent Layer (PPOAgentModel): Implements trading strategy and training logic
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
streamlit run main.py

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
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from utils.callbacks import ProgressBarCallback
from core.ppo_fin_model import PPOAgentModel
import tensorflow as tf
from tensorflow.keras import layers

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
        - model: PPOAgentModel - Trading agent model instance

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
        st.session_state.model = PPOAgentModel()


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


def main() -> None:
    """
    Main application entry point that sets up the Streamlit interface.

    System Components:
        1. Logging System:
           - Custom StreamlitLogHandler for UI integration
           - Configurable log levels and formatting

        2. UI Components:
           - Parameter input widgets
           - Training controls
           - Performance metrics display
           - Interactive charts

        3. Training Interface:
           - Model configuration
           - Progress tracking
           - Results visualization

        4. Testing Interface:
           - Model evaluation
           - Performance analysis
           - Portfolio tracking

    Implementation Flow:
        1. Initialize session state and logging
        2. Setup UI components and layouts
        3. Handle user interactions and model training
        4. Process and display results

    Error Handling:
        - Graceful handling of training failures
        - User feedback for invalid inputs
        - Session state persistence

    Example Usage:
        ```bash
        streamlit run main.py
        ```
    """
    init_session_state()

    # Configure logging
    handler = StreamlitLogHandler()
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    # Create sidebar for logs
    with st.sidebar:
        st.header("Logs")
        for log in st.session_state.log_messages:
            st.text(log)

    st.title("Trading Agent Configuration")

    # Input parameters
    st.header("Test Options")
    stock_name = st.text_input("Stock Name", value="AAPL")

    # Environment parameters
    st.header("Environment Parameters")
    col1, col2 = st.columns(2)
    with col1:
        initial_balance = st.number_input("Initial Balance", value=10000)

    with col2:
        transaction_cost = st.number_input("Transaction Cost",
                                           value=0.01,
                                           step=0.001)

    # Agent parameters
    st.header("Agent Parameters")
    col3, col4 = st.columns(2)
    with col3:
        learning_rate = st.number_input("Learning Rate",
                                        value=3e-4,
                                        format="%.1e")
        n_steps = st.number_input("Number of Steps", value=512)
        batch_size = st.number_input("Batch Size", value=128)
        n_epochs = st.number_input("Number of Epochs", value=5)
    with col4:
        gamma = st.number_input("Gamma (Discount Factor)", value=0.99)
        clip_range = st.number_input("Clip Range", value=0.2)
        target_kl = st.number_input("Target KL Divergence", value=0.05)

    col_train, col_test = st.columns(2)

    # Date selection
    st.subheader("Training Period")
    train_col1, train_col2 = st.columns(2)
    with train_col1:
        train_start_date = st.date_input("Training Start Date",
                                         value=datetime.now() -
                                         timedelta(days=365 * 5))
    with train_col2:
        train_end_date = st.date_input("Training End Date",
                                       value=datetime.now() -
                                       timedelta(days=365 + 1))

    if col_train.button("Start Training"):
        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        env_params = {
            'initial_balance': initial_balance,
            'transaction_cost': transaction_cost,
            'use_position_profit': False,
            'use_holding_bonus': False,
            'use_trading_penalty': False
        }

        ppo_params = {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'clip_range': clip_range,
            'target_kl': target_kl
        }

        progress_callback = ProgressBarCallback(
            total_timesteps=(train_end_date - train_start_date).days,
            progress_bar=progress_bar,
            status_placeholder=status_placeholder)

        metrics = st.session_state.model.train(stock_name=stock_name,
                                               start_date=train_start_date,
                                               end_date=train_end_date,
                                               env_params=env_params,
                                               ppo_params=ppo_params,
                                               callback=progress_callback)

        if metrics:
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                st.metric("Maximum Drawdown", f"{metrics['max_drawdown']:.2%}")
            with metrics_col2:
                st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
                st.metric("Volatility", f"{metrics['volatility']:.2%}")
            with metrics_col3:
                st.metric("Total Return", f"{metrics['total_return']:.2%}")
                st.metric("Final Portfolio Value",
                          f"${metrics['final_value']:,.2f}")

        st.success("Training completed and model saved!")

    # Test period dates
    st.subheader("Test Period")
    test_col1, test_col2 = st.columns(2)
    with test_col1:
        test_start_date = st.date_input("Test Start Date",
                                        value=datetime.now() -
                                        timedelta(days=365))
    with test_col2:
        test_end_date = st.date_input("Test End Date", value=datetime.now())

    if col_test.button("Test Model"):
        try:
            if not os.path.exists("trained_model.zip"):
                st.error("Please train the model first before testing!")
                return

            env_params = {
                'initial_balance': initial_balance,
                'transaction_cost': transaction_cost,
            }

            test_results = st.session_state.model.test(
                stock_name=stock_name,
                start_date=test_start_date,
                end_date=test_end_date,
                env_params=env_params,
                ppo_params=st.session_state.ppo_params)

            with st.expander("Test Results", expanded=True):
                progress_bar = st.progress(0)
                metrics_placeholder = st.empty()

                col1, col2, col3 = st.columns(3)
                metrics = test_results['metrics']

                with col1:
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    st.metric("Maximum Drawdown",
                              f"{metrics['max_drawdown']:.2%}")
                    st.metric(
                        "Total Return",
                        f"{(test_results['portfolio_history'][-1] - test_results['portfolio_history'][0]) / test_results['portfolio_history'][0]:.2%}"
                    )

                with col2:
                    st.metric("Sortino Ratio",
                              f"{metrics['sortino_ratio']:.2f}")
                    st.metric("Information Ratio",
                              f"{metrics['information_ratio']:.2f}")
                    st.metric("Volatility", f"{metrics['volatility']:.2%}")

                with col3:
                    st.metric("Beta", f"{metrics['beta']:.2f}")
                    st.metric(
                        "Final Portfolio Value",
                        f"${test_results['portfolio_history'][-1]:,.2f}")
                    st.metric("Initial Balance",
                              f"${test_results['portfolio_history'][0]:,.2f}")

                # Plot portfolio value over time
                st.subheader("Portfolio Value Over Time")
                st.line_chart(
                    pd.DataFrame(test_results['portfolio_history'],
                                 columns=['Portfolio Value']))

                # Create columns for charts
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    if len(test_results['returns']) > 0:
                        fig = go.Figure(data=[
                            go.Histogram(x=test_results['returns'], nbinsx=50)
                        ])
                        fig.update_layout(title="Returns Distribution",
                                          xaxis_title="Return",
                                          yaxis_title="Frequency",
                                          showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)

                    values = np.array(test_results['portfolio_history'])
                    peak = np.maximum.accumulate(values)
                    drawdowns = (peak - values) / peak
                    st.subheader("Drawdown Over Time")
                    st.area_chart(pd.DataFrame(drawdowns,
                                               columns=['Drawdown']))

                with chart_col2:
                    st.subheader("Cumulative Returns")
                    cum_returns = pd.DataFrame(
                        np.cumprod(1 + test_results['returns']) - 1,
                        columns=['Returns'])
                    st.line_chart(cum_returns)

                    st.subheader("30-Day Rolling Volatility")
                    rolling_vol = pd.DataFrame(
                        test_results['returns'],
                        columns=['Returns']).rolling(30).std() * np.sqrt(252)
                    st.line_chart(rolling_vol)

        except Exception as e:
            st.error(f"Error during testing: {str(e)}")


if __name__ == "__main__":
    main()
