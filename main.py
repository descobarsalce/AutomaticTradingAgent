
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
try:
    from core.trading_agent import TradingAgent
    from data.data_handler import DataHandler
    from core.visualization import TradingVisualizer
    from models.database import Session, StockData
except Exception as e:
    st.error(f"Import Error: {str(e)}")

st.set_page_config(
    page_title="RL Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_handler' not in st.session_state:
    st.session_state.data_handler = DataHandler()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = TradingVisualizer()

# Main page content
st.title("RL Trading Platform")
st.write("Welcome to the Trading Platform. Use the sidebar to navigate to different pages.")

# Quick stats
try:
    session = Session()
    total_symbols = session.query(StockData.symbol).distinct().count()
    latest_update = session.query(StockData.last_updated).order_by(StockData.last_updated.desc()).first()
    
    col1, col2 = st.columns(2)
    col1.metric("Total Symbols", total_symbols)
    if latest_update:
        col2.metric("Last Update", latest_update[0].strftime("%Y-%m-%d %H:%M"))
    session.close()
except Exception as e:
    st.error(f"Database connection error: {str(e)}")

st.markdown("""
## Navigation
- **Database Explorer**: View and analyze stored market data
- **Performance Plots**: Visualize trading performance and indicators
""")
