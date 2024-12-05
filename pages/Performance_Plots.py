import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from visualization import TradingVisualizer
from data_handler import DataHandler

# Page config
st.set_page_config(
    page_title="Performance Plots",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_handler' not in st.session_state:
    st.session_state.data_handler = DataHandler()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = TradingVisualizer()

# Page title
st.title("Performance Plots")

# Sidebar controls
st.sidebar.header("Chart Controls")

# Symbol selection
default_symbols = "AAPL,MSFT,GOOGL"
symbols_input = st.sidebar.text_input("Stock Symbols (comma-separated)", value=default_symbols)
symbols = [s.strip() for s in symbols_input.split(",")]

# Date range selection
end_date = datetime.now()
start_date = end_date - timedelta(days=30)  # Default to last 30 days
start_date = st.sidebar.date_input("Start Date", value=start_date)
end_date = st.sidebar.date_input("End Date", value=end_date)

# RSI Period selection
rsi_period = st.sidebar.slider("RSI Period", min_value=7, max_value=21, value=14)

if st.sidebar.button("Generate Charts"):
    with st.spinner("Fetching and processing data..."):
        # Update RSI period in visualizer
        st.session_state.visualizer.rsi_period = rsi_period
        
        # Fetch data for selected symbols
        portfolio_data = st.session_state.data_handler.fetch_data(symbols, start_date, end_date)
        portfolio_data = st.session_state.data_handler.prepare_data()
        
        # Generate charts
        figs = st.session_state.visualizer.create_charts(portfolio_data)
        
        # Display charts for each symbol
        for symbol, fig in figs.items():
            st.subheader(f"{symbol} Technical Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display basic statistics
            if symbol in portfolio_data:
                data = portfolio_data[symbol]
                col1, col2, col3 = st.columns(3)
                
                # Price metrics
                latest_close = data['Close'].iloc[-1]
                price_change = (latest_close - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
                
                col1.metric(
                    "Latest Close",
                    f"${latest_close:.2f}",
                    f"{price_change:.2f}%"
                )
                
                # Volume metrics
                avg_volume = data['Volume'].mean()
                volume_change = (data['Volume'].iloc[-1] - avg_volume) / avg_volume * 100
                
                col2.metric(
                    "Average Volume",
                    f"{avg_volume:,.0f}",
                    f"{volume_change:.2f}%"
                )
                
                # RSI metrics
                latest_rsi = data['RSI'].iloc[-1]
                rsi_change = latest_rsi - data['RSI'].iloc[-2]
                
                col3.metric(
                    "Current RSI",
                    f"{latest_rsi:.2f}",
                    f"{rsi_change:.2f}"
                )

# Instructions
if 'portfolio_data' not in locals():
    st.info("Select symbols and date range, then click 'Generate Charts' to view performance plots.")
