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
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")

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
        data = st.session_state.data_handler.fetch_data(symbol, start_date, end_date)
        data = st.session_state.data_handler.prepare_data()
        
        # Create environment and agent
        env = TradingEnvironment(data, initial_balance)
        agent = TradingAgent(env)
        
        # Train agent
        with st.spinner("Training agent..."):
            agent.train(training_steps)
            
        # Run evaluation episode
        obs = env.reset()
        done = False
        trades = []
        
        while not done:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            
            if abs(action[0]) > 0.1:  # Record significant trades
                trades.append({
                    'timestamp': data.index[env.current_step],
                    'action': action[0],
                    'price': data.iloc[env.current_step]['Close']
                })
        
        trades_df = pd.DataFrame(trades).set_index('timestamp')
        
        # Visualize results
        fig = st.session_state.visualizer.create_chart(data)
        st.session_state.visualizer.add_trades(trades_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display performance metrics
        col1, col2, col3 = st.columns(3)
        
        final_balance = env.net_worth
        returns = (final_balance - initial_balance) / initial_balance * 100
        
        col1.metric("Final Balance", f"${final_balance:,.2f}")
        col2.metric("Returns", f"{returns:.2f}%")
        col3.metric("Number of Trades", len(trades))

# Instructions
if 'data' not in locals():
    st.info("Enter a stock symbol and click 'Fetch Data & Train' to begin.")
