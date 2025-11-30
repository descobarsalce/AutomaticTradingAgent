"""
Hyperparameter Tuning Component
Handles the hyperparameter tuning interface
"""
import streamlit as st
import logging
from datetime import datetime, timedelta

from utils.stock_utils import parse_stock_list
from core.hyperparameter_search import hyperparameter_tuning

logger = logging.getLogger(__name__)


def display_tuning_tab():
    """
    Renders the hyperparameter tuning interface tab
    """
    start_time = datetime.now()
    logger.info("Initializing hyperparameter tuning tab...")
    st.header("Hyperparameter Tuning Configuration")

    # Add a checkbox for enabling/disabling logging
    enable_logging = st.checkbox("Enable Logging", value=False, key="tuning_enable_logging")
    st.session_state.enable_logging = enable_logging

    # Set the logging level based on the checkbox
    if enable_logging:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.CRITICAL)

    # Input parameters
    st.subheader("Training Options")
    stock_names = st.text_input("Training Stock Symbol",
                                value="AAPL,MSFT,TSLA,GOOG,NVDA",
                                key="tuning_stock_names")
    st.session_state.stock_names = parse_stock_list(stock_names)

    # Environment parameters
    st.header("Environment Parameters")
    col1, col2 = st.columns(2)
    with col1:
        initial_balance = st.number_input("Initial Balance", value=10000, key="tuning_initial_balance")

    with col2:
        transaction_cost = st.number_input("Transaction Cost",
                                           value=0.01,
                                           step=0.001,
                                           key="tuning_transaction_cost")

    st.session_state.env_params = {
        'initial_balance': initial_balance,
        'transaction_cost': transaction_cost,
        'max_pct_position_by_asset': 0.5,
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
                          value=datetime.now() - timedelta(days=365 * 5),
                          key="tuning_train_start_date"),
            datetime.min.time())
    with train_col2:
        train_end_date = datetime.combine(
            st.date_input("Training End Date",
                          value=datetime.now() - timedelta(days=365 + 1),
                          key="tuning_train_end_date"),
            datetime.min.time())

    st.session_state.train_start_date = train_start_date
    st.session_state.train_end_date = train_end_date

    # Hyperparameter tuning section
    hyperparameter_tuning()
