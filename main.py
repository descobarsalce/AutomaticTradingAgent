import streamlit as st
import os
import logging
from datetime import datetime, timedelta

from utils.callbacks import ProgressBarCallback
from metrics.metrics_calculator import MetricsCalculator
from environment import SimpleTradingEnv
from core import TradingAgent
import pandas as pd

def init_session_state():
    """Initialize all session state variables"""
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    if 'ppo_params' not in st.session_state:
        st.session_state.ppo_params = None

class StreamlitLogHandler(logging.Handler):
    def emit(self, record):
        try:
            log_entry = self.format(record)
            if 'log_messages' in st.session_state:
                st.session_state.log_messages.append(log_entry)
                if len(st.session_state.log_messages) > 100:
                    st.session_state.log_messages = st.session_state.log_messages[-100:]
            print(log_entry)  # Also print to console
        except Exception as e:
            print(f"Logging error: {e}")

def main():
    # Initialize session state first
    init_session_state()

    # Configure logging
    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    # Set up logging handler
    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    # Create sidebar for logs
    with st.sidebar:
        st.header("Logs")
        for log in st.session_state.log_messages:
            st.text(log)

    st.title("Trading Agent Configuration")
    
    # Rest of your main function implementation...
    # (keeping all the existing functionality)
    
    # Reward component controls
    st.header("Reward Components")
    use_position_profit = st.checkbox("Include Position Profit", value=False)
    use_holding_bonus = st.checkbox("Include Holding Bonus", value=False) 
    use_trading_penalty = st.checkbox("Include Trading Penalty", value=False)

    # Training mode parameters
    st.header("Training Mode")
    quick_mode = st.checkbox("Quick Training Mode", value=False)
    fast_eval = st.checkbox("Fast Evaluation", value=False)
    log_frequency = st.number_input("Log Frequency (steps)", value=50, min_value=1, help="How often to log portfolio state (in steps)")

    # Environment parameters
    st.header("Environment Parameters")
    col1, col2 = st.columns(2)
    with col1:
        initial_balance = st.number_input("Initial Balance", value=10000)
        transaction_cost = st.number_input("Transaction Cost", value=0.0, step=0.001)
        min_transaction_size = st.number_input("Minimum Transaction Size", value=0.001, step=0.001)
    with col2:
        max_position_pct = st.number_input("Maximum Position %", value=0.95, min_value=0.0, max_value=1.0)
        
    # Agent parameters
    st.header("Agent Parameters")
    col3, col4 = st.columns(2)
    with col3:
        learning_rate = st.number_input("Learning Rate", value=3e-4, format="%.1e", help="Suggested range: 1e-5 to 1e-3. Default: 3e-4")
        n_steps = st.number_input("Number of Steps", value=512, help="Suggested range: 64 to 2048. Default: 256")
        batch_size = st.number_input("Batch Size", value=128, help="Suggested range: 32 to 512. Default: 128")
        n_epochs = st.number_input("Number of Epochs", value=3, help="Suggested range: 1 to 10. Default: 3")
    with col4:
        gamma = st.number_input("Gamma (Discount Factor)", value=0.99, help="Suggested range: 0.8 to 0.999. Default: 0.99")
        clip_range = st.number_input("Clip Range", value=0.2, help="Suggested range: 0.0 to 0.5. Default: 0.2")
        target_kl = st.number_input("Target KL Divergence", value=0.05, help="Suggested range: 0.01 to 0.1. Default: 0.05")

    # Add test buttons
    test_mode = st.checkbox("Test Mode (100 steps)", value=False)
    
    col_train, col_test = st.columns(2)
    
    # Initialize database session
    from models.database import Session
    from models.models import StockData
    
    session = Session()

    # Add date selection for training period
    st.subheader("Training Period")
    train_col1, train_col2 = st.columns(2)
    with train_col1:
        train_start_date = st.date_input("Training Start Date", value=datetime.now() - timedelta(days=365))
    with train_col2:
        train_end_date = st.date_input("Training End Date", value=datetime.now() - timedelta(days=30))
        
    # Use DataHandler to get and prepare data with features
    from data.data_handler import DataHandler
    data_handler = DataHandler()

    if col_train.button("Start Training"):
        # Create progress tracking elements
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        
        
        portfolio_data = data_handler.fetch_data(symbols=['AAPL'], start_date=train_start_date, end_date=train_end_date)
        
        if not portfolio_data:
            st.error("No data found in database. Please add some stock data first.")
            return
            
        # Prepare data with technical indicators
        prepared_data = data_handler.prepare_data()
        
        # Use the first symbol's data since environment expects a single DataFrame
        data = next(iter(prepared_data.values()))
        
        if len(data) < 50:
            st.error("Insufficient data points (minimum 50 required for technical indicators)")
            return
        
        # Create environment with selected components
        env = SimpleTradingEnv(
            data=data,
            initial_balance=initial_balance,
            transaction_cost=transaction_cost,
            min_transaction_size=min_transaction_size,
            max_position_pct=max_position_pct,
            use_position_profit=use_position_profit,
            use_holding_bonus=use_holding_bonus,
            use_trading_penalty=use_trading_penalty,
            training_mode=True,
            log_frequency=log_frequency
        )
        
        # Configure PPO parameters
        st.session_state.ppo_params = {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'clip_range': clip_range,
            'target_kl': target_kl
        }
        
        # Initialize and train agent
        agent = TradingAgent(
            env=env,
            ppo_params=st.session_state.ppo_params,
            quick_mode=quick_mode,
            fast_eval=fast_eval
        )
        
        # Set timesteps based on test mode
        total_timesteps = 100 if test_mode else 10000
        
        # Create progress callback
        progress_callback = ProgressBarCallback(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
            status_placeholder=status_placeholder
        )
        
        agent.train(total_timesteps=total_timesteps, callback=progress_callback)
        agent.save("trained_model.zip")
        
        # Calculate and display performance metrics
        portfolio_history = env.get_portfolio_history()
        if len(portfolio_history) > 1:
            returns = MetricsCalculator.calculate_returns(portfolio_history)
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("Sharpe Ratio", 
                         f"{MetricsCalculator.calculate_sharpe_ratio(returns):.2f}")
                st.metric("Maximum Drawdown", 
                         f"{MetricsCalculator.calculate_maximum_drawdown(portfolio_history):.2%}")
                
            with metrics_col2:
                st.metric("Sortino Ratio",
                         f"{MetricsCalculator.calculate_sortino_ratio(returns):.2f}")
                st.metric("Volatility",
                         f"{MetricsCalculator.calculate_volatility(returns):.2%}")
                
            with metrics_col3:
                final_value = portfolio_history[-1]
                initial_value = portfolio_history[0]
                total_return = (final_value - initial_value) / initial_value
                st.metric("Total Return", f"{total_return:.2%}")
                st.metric("Final Portfolio Value", f"${final_value:,.2f}")
                
        st.success("Training completed and model saved!")
        
    # Test period dates
    st.subheader("Test Period")
    test_col1, test_col2 = st.columns(2)
    with test_col1:
        test_start_date = st.date_input("Test Start Date", value=datetime.now() - timedelta(days=30))
    with test_col2:
        test_end_date = st.date_input("Test End Date", value=datetime.now())
        
    if col_test.button("Test Model"):
        try:
            model_path = "trained_model.zip"
            if not os.path.exists(model_path):
                st.error("Please train the model first before testing!")
                return
                
            # Create test environment and data with selected dates
            data_handler = DataHandler()
            test_portfolio_data = data_handler.fetch_data(symbols=['AAPL'], start_date=test_start_date, end_date=test_end_date)
            test_prepared_data = data_handler.prepare_data()
            test_data = next(iter(test_prepared_data.values()))[-100:]  # Get last 100 records with features
            
            test_env = SimpleTradingEnv(
                data=test_data,
                initial_balance=initial_balance,
                transaction_cost=transaction_cost,
                min_transaction_size=min_transaction_size,
                max_position_pct=max_position_pct,
                use_position_profit=use_position_profit,
                use_holding_bonus=use_holding_bonus,
                use_trading_penalty=use_trading_penalty
            )
            
            # Initialize agent with test environment
            test_agent = TradingAgent(
                env=test_env,
                ppo_params=st.session_state.ppo_params,
                quick_mode=quick_mode,
                fast_eval=fast_eval
            )
            
            # Create log display area
            log_container = st.expander("Test Logs", expanded=True)
            log_placeholder = log_container.empty()
            
            test_agent.load("trained_model.zip")
            
            # Run test episode
            obs, _ = test_env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            with st.expander("Test Results", expanded=True):
                progress_bar = st.progress(0)
                metrics_placeholder = st.empty()
                
                total_steps = len(test_data)
                while not done:
                    action = test_agent.predict(obs)
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    steps += 1
                    
                    # Update progress and metrics
                    progress_bar.progress(steps / total_steps)
                    metrics_placeholder.write({
                        'Step': steps,
                        'Reward': round(total_reward, 2),
                        'Portfolio Value': round(info['net_worth'], 2),
                        'Position': round(float(test_env.shares_held), 3)
                    })
                
                st.success(f"Test completed! Final portfolio value: ${info['net_worth']:.2f}")
                
        except Exception as e:
            st.error(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    main()