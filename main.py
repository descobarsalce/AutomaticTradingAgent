
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from models.database import Session
from models.models import StockData

st.set_page_config(
    page_title="RL Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Initialize session state variables safely"""
    if 'initialized' not in st.session_state:
        try:
            from data.data_handler import DataHandler
            from core.visualization import TradingVisualizer
            st.session_state.data_handler = DataHandler()
            st.session_state.visualizer = TradingVisualizer()
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Failed to initialize components: {str(e)}")
            st.session_state.initialized = False

def main():
    st.title("RL Trading Platform")
    st.write("Welcome to the Trading Platform. Use the sidebar to navigate to different pages.")
    
    init_session_state()
    
    if not st.session_state.get('initialized', False):
        st.warning("System initialization failed. Please check your installation.")
        return
        
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

if __name__ == "__main__":
    main()
