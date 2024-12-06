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

# Data loading
if st.sidebar.button("Fetch Data & Train"):
    with st.spinner("Fetching data..."):
        portfolio_data = st.session_state.data_handler.fetch_data(symbols, start_date, end_date)
        portfolio_data = st.session_state.data_handler.prepare_data()
        
        # Create environments and agents for each stock
        environments = {}
        agents = {}
        all_trades = {}
        
        for symbol, data in portfolio_data.items():
            # Allocate initial balance based on weights
            symbol_balance = initial_balance * weights[symbol]
            environments[symbol] = TradingEnvironment(
                data=data,
                initial_balance=symbol_balance
            )
            
            # Debug logging for action space verification
            print(f"Environment action space for {symbol}:", environments[symbol].action_space)
            
            # Initialize agent with environment
            agents[symbol] = TradingAgent(environments[symbol])
            
            # Train agent
            with st.spinner(f"Training agent for {symbol}..."):
                agents[symbol].train(training_steps)
                
            # Run evaluation episode
            obs, info = environments[symbol].reset()
            done = False
            trades = []
            
            while not done:
                action = agents[symbol].predict(obs)
                obs, reward, done, info = environments[symbol].step(action)
                
                if abs(action[0]) > 0.1:  # Record significant trades
                    trades.append({
                        'timestamp': data.index[environments[symbol].current_step],
                        'action': action[0],
                        'price': data.iloc[environments[symbol].current_step]['Close']
                    })
                
                # Update agent's state tracking
                agents[symbol].update_state(
                    portfolio_value=info['net_worth'],
                    positions={symbol: info['shares_held']}
                )
            
            all_trades[symbol] = pd.DataFrame(trades).set_index('timestamp')
        
        # Visualize results
        figs = st.session_state.visualizer.create_charts(portfolio_data, all_trades)
        
        # Display charts
        for symbol, fig in figs.items():
            st.subheader(f"{symbol} Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display performance metrics
            col1, col2, col3 = st.columns(3)
            
            final_balance = environments[symbol].net_worth
            symbol_initial_balance = initial_balance * weights[symbol]
            returns = (final_balance - symbol_initial_balance) / symbol_initial_balance * 100
            
            col1.metric(f"{symbol} Final Balance", f"${final_balance:,.2f}")
            col2.metric(f"{symbol} Returns", f"{returns:.2f}%")
            col3.metric(f"{symbol} Number of Trades", len(all_trades[symbol]))
            
        # Display portfolio summary
        st.subheader("Portfolio Summary")
        total_value = sum(env.net_worth for env in environments.values())
        portfolio_return = (total_value - initial_balance) / initial_balance * 100
        
        col1, col2 = st.columns(2)
        col1.metric("Total Portfolio Value", f"${total_value:,.2f}")
        col2.metric("Portfolio Return", f"{portfolio_return:.2f}%")

# Instructions
if 'data' not in locals():
    st.info("Enter a stock symbol and click 'Fetch Data & Train' to begin.")
