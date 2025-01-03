import streamlit as st
import os

from utils.callbacks import ProgressBarCallback

from environment import SimpleTradingEnv
from core import TradingAgent
import pandas as pd

def main():
    # Initialize session state for PPO parameters
    if 'ppo_params' not in st.session_state:
        st.session_state.ppo_params = None
        
    st.title("Trading Agent Configuration")
    
    # Reward component controls
    st.header("Reward Components")
    use_position_profit = st.checkbox("Include Position Profit", value=False)
    use_holding_bonus = st.checkbox("Include Holding Bonus", value=False)
    use_trading_penalty = st.checkbox("Include Trading Penalty", value=False)
    
    # Training mode parameters
    st.header("Training Mode")
    quick_mode = st.checkbox("Quick Training Mode", value=False)
    fast_eval = st.checkbox("Fast Evaluation", value=False)
    
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
        learning_rate = st.number_input("Learning Rate", value=3e-4, min_value=1e-5, max_value=1e-3, format="%.1e")
        n_steps = st.number_input("Number of Steps", value=256, min_value=64, max_value=2048)
        batch_size = st.number_input("Batch Size", value=128, min_value=32, max_value=512)
        n_epochs = st.number_input("Number of Epochs", value=3, min_value=1, max_value=10)
    with col4:
        gamma = st.number_input("Gamma (Discount Factor)", value=0.99, min_value=0.8, max_value=0.999)
        clip_range = st.number_input("Clip Range", value=0.2, min_value=0.1, max_value=0.5)
        target_kl = st.number_input("Target KL Divergence", value=0.05, min_value=0.01, max_value=0.1)
        
    # Add test buttons
    test_mode = st.checkbox("Test Mode (100 steps)", value=False)
    
    col_train, col_test = st.columns(2)
    
    if col_train.button("Start Training"):
        # Create progress tracking elements
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        
        # Create sample data for testing
        data = pd.DataFrame({
            'Open': [100] * 1000,
            'High': [110] * 1000,
            'Low': [90] * 1000,
            'Close': [105] * 1000,
            'Volume': [1000] * 1000
        })
        
        # Create environment with selected components
        env = SimpleTradingEnv(
            data=data,
            initial_balance=initial_balance,
            transaction_cost=transaction_cost,
            min_transaction_size=min_transaction_size,
            max_position_pct=max_position_pct,
            use_position_profit=use_position_profit,
            use_holding_bonus=use_holding_bonus,
            use_trading_penalty=use_trading_penalty
        )
        
        # Configure PPO parameters and store in session state
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
        # Save the trained model
        agent.save("trained_model.zip")
        st.success("Training completed and model saved!")
        
    if col_test.button("Test Model"):
        try:
            # Load the trained model
            model_path = "trained_model.zip"
            if not os.path.exists(model_path):
                st.error("Please train the model first before testing!")
                return
                
            # Create test environment and data
            test_data = pd.DataFrame({
                'Open': [100] * 100,
                'High': [110] * 100,
                'Low': [90] * 100,
                'Close': [105] * 100,
                'Volume': [1000] * 100
            })
            
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
        except Exception as e:
            st.error(f"Error during test setup: {str(e)}")
            return
        
        # Initialize agent with test environment and load trained weights
        test_agent = TradingAgent(
            env=test_env,
            ppo_params=st.session_state.ppo_params,
            quick_mode=quick_mode,
            fast_eval=fast_eval
        )
        try:
            test_agent.load("trained_model.zip")
        except Exception as e:
            st.error(f"Error loading the model: {str(e)}")
            return
        
        # Run test episode
        obs, _ = test_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        with st.expander("Test Results", expanded=True):
            progress_bar = st.progress(0)
            metrics_placeholder = st.empty()
            
            while not done and steps < 100:
                action = test_agent.predict(obs)
                obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                
                # Update progress and metrics
                progress_bar.progress(steps / 100)
                current_shares = test_env.shares_held  # Get shares directly from environment
                metrics_placeholder.write({
                    'Step': steps,
                    'Reward': round(total_reward, 2),
                    'Portfolio Value': round(info['net_worth'], 2),
                    'Position': round(float(current_shares), 3)
                })
            
            st.success(f"Test completed! Final portfolio value: ${info['net_worth']:.2f}")

if __name__ == "__main__":
    main()