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
available_symbols = ["AAPL", "MSFT", "GOOGL"]  # Default available symbols
selected_symbol = st.sidebar.selectbox("Select Symbol", available_symbols, index=0)
symbols = [selected_symbol]  # Keep as list for compatibility with existing code

# Date range selection
end_date = datetime.now()
start_date = end_date - timedelta(days=30)  # Default to last 30 days
start_date = st.sidebar.date_input("Start Date", value=start_date)
end_date = st.sidebar.date_input("End Date", value=end_date)

# Indicator selection
st.sidebar.subheader("Indicators")
show_rsi = st.sidebar.checkbox("Show RSI", value=True)
show_sma20 = st.sidebar.checkbox("Show SMA 20", value=True)
show_sma50 = st.sidebar.checkbox("Show SMA 50", value=True)

# RSI Period selection (only show if RSI is enabled)
rsi_period = 14
if show_rsi:
    rsi_period = st.sidebar.slider("RSI Period", min_value=7, max_value=21, value=14)

if st.sidebar.button("Generate Charts"):
    with st.spinner("Fetching and processing data..."):
        try:
            # Update visualizer settings
            st.session_state.visualizer.rsi_period = rsi_period
            st.session_state.visualizer.show_rsi = show_rsi
            st.session_state.visualizer.show_sma20 = show_sma20
            st.session_state.visualizer.show_sma50 = show_sma50
            
            # Fetch data for selected symbols
            portfolio_data = st.session_state.data_handler.fetch_data(symbols, start_date, end_date)
            if not portfolio_data:
                st.error("No data available for the selected symbols and date range.")
                st.stop()
                
            portfolio_data = st.session_state.data_handler.prepare_data()
            if not portfolio_data:
                st.error("Error preparing data for visualization.")
                st.stop()
            
            # Generate charts with progress indication
            st.text("Generating interactive charts...")
            progress_bar = st.progress(0)
            figs = {}
            
            for i, (symbol, data) in enumerate(portfolio_data.items()):
                try:
                    fig = st.session_state.visualizer.create_single_chart(symbol, data)
                    if fig:
                        figs[symbol] = fig
                    else:
                        st.warning(f"Could not generate chart for {symbol}")
                except Exception as e:
                    st.error(f"Error generating chart for {symbol}: {str(e)}")
                progress_bar.progress((i + 1) / len(portfolio_data))
            
            if not figs:
                st.error("Could not generate any charts. Please check the data and try again.")
                st.stop()
            
            # Display charts for each symbol
            for symbol, fig in figs.items():
                st.subheader(f"{symbol} Technical Analysis")
                st.plotly_chart(fig, use_container_width=True, config={
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape']
                })
                
                # Calculate and display basic statistics
                if symbol in portfolio_data:
                    data = portfolio_data[symbol]
                    col1, col2, col3 = st.columns(3)
                    
                    try:
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
                        
                        # RSI metrics (only if RSI is enabled and available)
                        if show_rsi and 'RSI' in data.columns:
                            latest_rsi = data['RSI'].iloc[-1]
                            rsi_change = latest_rsi - data['RSI'].iloc[-2] if len(data) > 1 else 0
                            
                            col3.metric(
                                "Current RSI",
                                f"{latest_rsi:.2f}",
                                f"{rsi_change:.2f}"
                            )
                    except Exception as e:
                        st.error(f"Error calculating metrics for {symbol}: {str(e)}")
                        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Instructions
if 'portfolio_data' not in locals():
    st.info("Select symbols and date range, then click 'Generate Charts' to view performance plots.")
