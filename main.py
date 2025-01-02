
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import logging
from datetime import datetime, timedelta

# Filter out torch warning messages
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
logging.getLogger('torch').setLevel(logging.ERROR)
from models.database import Session
from models.models import StockData
from environment import SimpleTradingEnv
from core import TradingAgent
from utils.callbacks import ProgressBarCallback

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
            data = pd.read_sql(f"""
                SELECT date, open as Open, high as High, low as Low, 
                       close as Close, volume as Volume 
                FROM stock_data 
                WHERE symbol = '{selected_symbol}' 
                ORDER BY date""", session.bind)
            
            # Ensure all required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                st.error("Missing required price data columns")
                return
                
            # Parameter Configuration Section
            st.subheader("Model Parameters")
            
            with st.expander("Environment Parameters", expanded=True):
                col1, col2 = st.columns(2)
                initial_balance = col1.number_input("Initial Balance ($)", min_value=1000, value=10000, help="Starting capital for trading")
                transaction_cost = col2.number_input("Transaction Cost", min_value=0.0, max_value=0.1, value=0.0, format="%f", help="Trading fee as a fraction (e.g., 0.001 = 0.1%)")
                
                col3, col4 = st.columns(2)
                min_transaction_size = col3.number_input("Minimum Transaction Size", min_value=0.0, value=0.001, help="Minimum shares per trade (can be fractional)")
                max_position_pct = col4.slider("Maximum Position Size (%)", min_value=10, max_value=100, value=95, help="Maximum portfolio % for single position") / 100.0

            with st.expander("Training Parameters", expanded=True):
                col1, col2, col3 = st.columns(3)
                epochs = col1.number_input("Training Epochs", min_value=1, value=10, help="Number of training cycles")
                learning_rate = col2.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=3e-4, format="%e", help="Model learning rate")
                n_steps = col3.number_input("Steps per Update", min_value=128, max_value=2048, value=256, help="Steps before model update")

                col4, col5, col6 = st.columns(3)
                batch_size = col4.number_input("Batch Size", min_value=32, max_value=512, value=256, help="Training batch size")
                gamma = col5.number_input("Gamma (Discount)", min_value=0.8, max_value=0.999, value=0.99, help="Future reward discount factor")
                quick_mode = col6.checkbox("Quick Mode", value=True, help="Enable faster training with simplified parameters")

            train_button = st.button("Train Agent")

            if train_button:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                env = SimpleTradingEnv(
                    data=data,
                    initial_balance=initial_balance,
                    transaction_cost=transaction_cost,
                    min_transaction_size=min_transaction_size,
                    max_position_pct=max_position_pct
                )
                
                ppo_params = {
                    'learning_rate': learning_rate,
                    'n_steps': n_steps,
                    'batch_size': batch_size,
                    'gamma': gamma
                }
                
                st.session_state.agent = TradingAgent(env, ppo_params=ppo_params, quick_mode=quick_mode)
                total_timesteps = epochs * len(env.data)
                
                # Create progress callback
                progress_callback = ProgressBarCallback(
                    total_timesteps=total_timesteps,
                    progress_bar=progress_bar,
                    status_placeholder=status_text
                )
                
                st.session_state.agent.train(total_timesteps=total_timesteps, callback=progress_callback)
                st.session_state.env = env
                st.success("Agent trained successfully!")
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
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def test_with_progress():
                        test_steps = len(st.session_state.env.data)
                        for step in range(test_steps):
                            progress = (step + 1) / test_steps
                            progress_bar.progress(progress)
                            status_text.text(f"Testing progress: {progress*100:.1f}%")
                            yield step
                            
                    for _ in test_with_progress():
                        pass
                        
                    reward, final_worth = test_agent(st.session_state.agent, st.session_state.env)
                    progress_bar.progress(1.0)
                    status_text.text("Testing completed!")
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
