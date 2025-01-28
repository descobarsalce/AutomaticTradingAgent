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
        from components.database_tab import display_database_explorer
        display_database_explorer()


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
