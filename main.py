
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from environment import TradingEnvironment, SimpleTradingEnv
from core.trading_agent import TradingAgent
from data.data_handler import DataHandler
from core.visualization import TradingVisualizer
from utils.callbacks import ProgressBarCallback, PortfolioMetricsCallback

if __name__ == "__main__":
    # Page config
    st.set_page_config(
        page_title="RL Trading Platform",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'data_handler' not in st.session_state:
        st.session_state.data_handler = DataHandler()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = TradingVisualizer()
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = None
    if 'trained_agents' not in st.session_state:
        st.session_state.trained_agents = {}
    if 'environments' not in st.session_state:
        st.session_state.environments = {}
    if 'all_trades' not in st.session_state:
        st.session_state.all_trades = {}
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False
    if 'data_validated' not in st.session_state:
        st.session_state.data_validated = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

    # Rest of your existing main.py code...
