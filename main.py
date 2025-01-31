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
from components.chatbot_tab import display_chatbot_tab

# Configure logging
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def init_session_state() -> None:
    """
    Initialize Streamlit session state variables for persistent storage across reruns.
    """
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    if 'ppo_params' not in st.session_state:
        st.session_state.ppo_params = None
    if 'model' not in st.session_state:
        st.session_state.model = UnifiedTradingAgent()
    if 'stock_list' not in st.session_state:
        st.session_state.stock_list = ['APPL', 'MSFT']

def main() -> None:
    init_session_state()

    st.title("Trading Analysis and Agent Platform")

    # Create tabs including the new Chatbot tab
    tab_training, tab_analysis, tab_database, tab_chatbot = st.tabs(
        ["Model Training", "Technical Analysis", "Database Explorer", "AI Assistant"])

    with tab_analysis:
        display_tech_analysis_tab()

    with tab_training:
        display_training_tab()

    with tab_database:
        display_database_explorer()

    with tab_chatbot:
        display_chatbot_tab()


if __name__ == "__main__":
    main()