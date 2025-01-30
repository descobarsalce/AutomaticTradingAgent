"""
Main application entry point. Provides interface for training and 
analyzing portfolio allocation trading strategies.
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
from models.database import Session, StockData
from utils.stock_utils import parse_stock_list

from components.analysis_tab import display_tech_analysis_tab
from components.training_tab import display_training_tab
from components.database_tab import display_database_explorer

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
        - stock_list: List[str] - List of stock symbols for portfolio

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
        st.session_state.model = UnifiedTradingAgent(optimize_for_sharpe=True)
    if 'stock_list' not in st.session_state:
        st.session_state.stock_list = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']

def main() -> None:
    """Main application entry point."""
    init_session_state()

    st.title("Portfolio Allocation and Trading Platform")

    # Create tabs for different functionalities
    tab_training, tab_analysis, tab_database = st.tabs(
        ["Model Training", "Technical Analysis", "Database Explorer"])

    # Display appropriate content for each tab
    with tab_analysis:
        display_tech_analysis_tab()

    with tab_training:
        display_training_tab()

    with tab_database:
        display_database_explorer()


if __name__ == "__main__":
    main()