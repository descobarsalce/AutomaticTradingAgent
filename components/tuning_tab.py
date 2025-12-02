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
    st.header("Hyperparameter Tuning")

    # Compact settings row
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        stock_names = st.text_input(
            "Stocks", value="AAPL,MSFT,TSLA,GOOG,NVDA", key="tuning_stock_names",
            help="Comma-separated stock symbols"
        )
        st.session_state.stock_names = parse_stock_list(stock_names)
    with col2:
        initial_balance = st.number_input(
            "Balance ($)", value=10000, key="tuning_initial_balance", format="%d"
        )
    with col3:
        transaction_cost = st.number_input(
            "Trans. Cost", value=0.01, step=0.001, key="tuning_transaction_cost", format="%.3f"
        )
    with col4:
        enable_logging = st.checkbox("Logging", value=False, key="tuning_enable_logging")
        st.session_state.enable_logging = enable_logging

    if enable_logging:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.CRITICAL)

    st.session_state.env_params = {
        'initial_balance': initial_balance,
        'transaction_cost': transaction_cost,
        'max_pct_position_by_asset': 0.5,
        'use_position_profit': False,
        'use_holding_bonus': False,
        'use_trading_penalty': False
    }

    # Training period - compact date row
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        train_start_date = datetime.combine(
            st.date_input("Start Date", value=datetime.now() - timedelta(days=365 * 5),
                          key="tuning_train_start_date"),
            datetime.min.time())
    with date_col2:
        train_end_date = datetime.combine(
            st.date_input("End Date", value=datetime.now() - timedelta(days=365 + 1),
                          key="tuning_train_end_date"),
            datetime.min.time())

    st.session_state.train_start_date = train_start_date
    st.session_state.train_end_date = train_end_date

    # Hyperparameter tuning section
    hyperparameter_tuning()
