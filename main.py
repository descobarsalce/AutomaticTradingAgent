
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models.database import Session
from models.models import StockData
from environment import SimpleTradingEnv
from core import TradingAgent

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
            st.session_state.agent = None
            st.session_state.env = None
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Failed to initialize components: {str(e)}")
            st.session_state.initialized = False

def train_agent(env, epochs, quick_mode=False):
    """Train the trading agent"""
    agent = TradingAgent(env, quick_mode=quick_mode)
    total_timesteps = epochs * len(env.data)
    agent.train(total_timesteps=total_timesteps)
    return agent

def test_agent(agent, env):
    """Test the trading agent"""
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    return total_reward, info['net_worth']

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

        # Agent Training Section
        st.header("Train Trading Agent")
        symbols = [symbol[0] for symbol in session.query(StockData.symbol).distinct().all()]
        selected_symbol = st.selectbox("Select Symbol", symbols)
        
        if selected_symbol:
            data = pd.read_sql(f"SELECT * FROM stock_data WHERE symbol = '{selected_symbol}' ORDER BY date", session.bind)
            
            col1, col2, col3 = st.columns(3)
            epochs = col1.number_input("Training Epochs", min_value=1, value=10)
            quick_mode = col2.checkbox("Quick Mode", value=True)
            train_button = col3.button("Train Agent")

            if train_button:
                with st.spinner("Training agent..."):
                    env = SimpleTradingEnv(data=data, initial_balance=10000)
                    st.session_state.agent = train_agent(env, epochs, quick_mode)
                    st.session_state.env = env
                    st.success("Agent trained successfully!")

            # Testing Section
            if st.session_state.agent is not None:
                st.header("Test Agent")
                test_button = st.button("Test Agent")
                
                if test_button:
                    with st.spinner("Testing agent..."):
                        reward, final_worth = test_agent(st.session_state.agent, st.session_state.env)
                        st.metric("Total Reward", f"{reward:.2f}")
                        st.metric("Final Portfolio Value", f"${final_worth:.2f}")

                # Display metrics if available
                metrics = st.session_state.agent.get_metrics()
                if metrics:
                    st.subheader("Agent Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    col2.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
                    col3.metric("Win Rate", f"{metrics['win_rate']:.2%}")

        session.close()
    except Exception as e:
        st.error(f"Error: {str(e)}")

    st.markdown("""
    ## Navigation
    - **Database Explorer**: View and analyze stored market data
    - **Performance Plots**: Visualize trading performance and indicators
    """)

if __name__ == "__main__":
    main()
