
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

from environment.trading import TradingEnvironment
from environment.simple_trading_env import SimpleTradingEnv
from core.trading_agent import TradingAgent
from data.data_handler import DataHandler
from core.visualization import TradingVisualizer
from utils.callbacks import ProgressBarCallback, PortfolioMetricsCallback
from metrics.metrics_calculator import MetricsCalculator
from models.database import Session, StockData

# Single page config
st.set_page_config(
    page_title="RL Trading Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rest of your main.py code...
