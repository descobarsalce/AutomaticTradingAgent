import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from environment import TradingEnvironment
from agent import TradingAgent
from data_handler import DataHandler
from visualization import TradingVisualizer

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

# Sidebar
st.sidebar.title("Trading Parameters")

# Symbol selection
default_symbols = "AAPL,MSFT,GOOGL"
symbols_input = st.sidebar.text_input("Stock Symbols (comma-separated)", value=default_symbols)
symbols = [s.strip() for s in symbols_input.split(",")]

# Portfolio weights
st.sidebar.subheader("Portfolio Weights")
weights = {}
for symbol in symbols:
    weights[symbol] = st.sidebar.slider(f"{symbol} Weight (%)", 0, 100, 100 // len(symbols)) / 100

# Date range
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=start_date)
end_date = st.sidebar.date_input("End Date", value=end_date)

# Training parameters
initial_balance = st.sidebar.number_input("Initial Balance ($)", value=100000)
training_steps = st.sidebar.number_input("Training Steps", value=10000, step=1000)

# Main content
st.title("Reinforcement Learning Trading Platform")

# Step 1: Data Fetching
fetch_data = st.sidebar.button("1. Fetch Market Data")
if fetch_data:
    with st.spinner("Fetching and preparing market data..."):
        try:
            st.session_state.portfolio_data = st.session_state.data_handler.fetch_data(symbols, start_date, end_date)
            st.session_state.portfolio_data = st.session_state.data_handler.prepare_data()
            st.success("✅ Market data fetched and prepared successfully!")
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            st.session_state.portfolio_data = None

# Show data preview if available
if st.session_state.portfolio_data is not None:
    st.subheader("Market Data Preview")
    for symbol, data in st.session_state.portfolio_data.items():
        with st.expander(f"{symbol} Data Preview"):
            st.dataframe(data.head())
            st.text(f"Total records: {len(data)}")

# Step 2: Data Validation
if st.session_state.portfolio_data is not None:
    st.subheader("Data Validation")
    
    for symbol, data in st.session_state.portfolio_data.items():
        with st.expander(f"{symbol} Data Validation"):
            # Calculate basic statistics
            missing_values = data.isnull().sum()
            data_completeness = (1 - missing_values / len(data)) * 100
            
            # Display data quality metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Data Points", len(data))
            col2.metric("Data Completeness", f"{data_completeness.mean():.2f}%")
            
            # Check for price anomalies
            price_std = data['Close'].std()
            price_mean = data['Close'].mean()
            anomalies = data[abs(data['Close'] - price_mean) > 3 * price_std]
            col3.metric("Price Anomalies", len(anomalies))
            
            # Display statistical summary
            st.write("Statistical Summary:")
            st.dataframe(data.describe())
            
            # Show missing values if any
            if missing_values.sum() > 0:
                st.warning("Missing Values Detected:")
                st.write(missing_values[missing_values > 0])
            
            # Visualize price distribution
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=data['Close'], name='Price Distribution'))
            fig.update_layout(title=f"{symbol} Price Distribution", xaxis_title="Price", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
            
            # Data quality warnings
            if len(anomalies) > 0:
                st.warning(f"Found {len(anomalies)} potential price anomalies")
            if data_completeness.mean() < 95:
                st.warning(f"Data completeness below 95% threshold")

# Step 3: Training
train_model = st.sidebar.button(
    "3. Train Model",
    disabled=st.session_state.portfolio_data is None
)

if train_model:
    st.session_state.environments = {}
    st.session_state.trained_agents = {}
    st.session_state.all_trades = {}
    
    progress_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    for idx, (symbol, data) in enumerate(st.session_state.portfolio_data.items()):
        progress = idx / len(st.session_state.portfolio_data)
        progress_placeholder.text(f"Processing {symbol}...")
        progress_bar.progress(progress)
        
        # Allocate initial balance based on weights
        symbol_balance = initial_balance * weights[symbol]
        st.session_state.environments[symbol] = TradingEnvironment(
            data=data,
            initial_balance=symbol_balance
        )
        
        # Initialize agent with environment
        st.session_state.trained_agents[symbol] = TradingAgent(st.session_state.environments[symbol])
        
        # Train agent with progress tracking
        training_progress = st.progress(0)
        training_status = st.empty()
        
        with st.spinner(f"Training agent for {symbol}..."):
            try:
                st.session_state.trained_agents[symbol].train(training_steps)
                training_progress.progress(1.0)
                training_status.text(f"Training completed for {symbol}")
            except Exception as e:
                st.error(f"Error training agent for {symbol}: {str(e)}")
                continue
            
        # Run evaluation episode
        obs, info = st.session_state.environments[symbol].reset()
        done = False
        trades = []
        
        eval_progress = st.progress(0)
        eval_status = st.empty()
        
        while not done:
            action = st.session_state.trained_agents[symbol].predict(obs)
            obs, reward, terminated, truncated, info = st.session_state.environments[symbol].step(action)
            done = terminated or truncated
            
            # Update evaluation progress
            progress = st.session_state.environments[symbol].current_step / len(data)
            eval_progress.progress(progress)
            eval_status.text(f"Evaluating {symbol}: Step {st.session_state.environments[symbol].current_step}")
            
            if abs(action[0]) > 0.1:  # Record significant trades
                try:
                    trade_data = {
                        'timestamp': data.index[st.session_state.environments[symbol].current_step],
                        'action': action[0],
                        'price': data.iloc[st.session_state.environments[symbol].current_step]['Close']
                    }
                    # Validate timestamp exists and is valid
                    if trade_data['timestamp'] is not None:
                        trades.append(trade_data)
                    else:
                        st.warning(f"Invalid timestamp for trade at step {st.session_state.environments[symbol].current_step}")
                except IndexError as e:
                    st.error(f"Error recording trade: Index out of bounds at step {st.session_state.environments[symbol].current_step}")
                    continue
                except Exception as e:
                    st.error(f"Unexpected error recording trade: {str(e)}")
                    continue
            
            # Update agent's state tracking
            st.session_state.trained_agents[symbol].update_state(
                portfolio_value=info['net_worth'],
                positions={symbol: info['shares_held']}
            )
        
        try:
            if trades:  # Verify trades list is not empty
                # Verify all trades have timestamp field
                if all('timestamp' in trade for trade in trades):
                    st.session_state.all_trades[symbol] = pd.DataFrame(trades).set_index('timestamp')
                else:
                    # Fallback: Create DataFrame without index if timestamps are missing
                    st.warning(f"Missing timestamps in trades for {symbol}. Using default index.")
                    st.session_state.all_trades[symbol] = pd.DataFrame(trades)
            else:
                # Handle case where no trades were made
                st.info(f"No significant trades recorded for {symbol}")
                st.session_state.all_trades[symbol] = pd.DataFrame(columns=['timestamp', 'action', 'price'])
        except Exception as e:
            st.error(f"Error creating trades DataFrame for {symbol}: {str(e)}")
            st.session_state.all_trades[symbol] = pd.DataFrame(columns=['timestamp', 'action', 'price'])
        
    st.session_state.training_completed = True

# Step 3: View Results
if st.session_state.training_completed:
    st.sidebar.success("✅ Training completed! View results below.")
    
    # Visualize results
    figs = st.session_state.visualizer.create_charts(
        st.session_state.portfolio_data, 
        st.session_state.all_trades
    )
    
    # Display charts
    for symbol, fig in figs.items():
        st.subheader(f"{symbol} Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Display performance metrics
        col1, col2, col3 = st.columns(3)
        
        final_balance = st.session_state.environments[symbol].net_worth
        symbol_initial_balance = initial_balance * weights[symbol]
        returns = (final_balance - symbol_initial_balance) / symbol_initial_balance * 100
        
        col1.metric(f"{symbol} Final Balance", f"${final_balance:,.2f}")
        col2.metric(f"{symbol} Returns", f"{returns:.2f}%")
        col3.metric(f"{symbol} Number of Trades", len(st.session_state.all_trades[symbol]))
        
    # Display portfolio summary
    st.subheader("Portfolio Summary")
    total_value = sum(env.net_worth for env in st.session_state.environments.values())
    portfolio_return = (total_value - initial_balance) / initial_balance * 100
    
    col1, col2 = st.columns(2)
    col1.metric("Total Portfolio Value", f"${total_value:,.2f}")
    col2.metric("Portfolio Return", f"{portfolio_return:.2f}%")
    
    # Add reset button
    if st.sidebar.button("Reset Session"):
        st.session_state.portfolio_data = None
        st.session_state.trained_agents = {}
        st.session_state.environments = {}
        st.session_state.all_trades = {}
        st.session_state.training_completed = False
        st.experimental_rerun()

# Instructions
if st.session_state.portfolio_data is None:
    st.info("Select symbols and click '1. Fetch Market Data' to begin.")
