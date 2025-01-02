
import streamlit as st
from environment import SimpleTradingEnv
from core import TradingAgent
import pandas as pd

def main():
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
        
    if st.button("Start Training"):
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
        
        # Configure PPO parameters
        ppo_params = {
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
            ppo_params=ppo_params,
            quick_mode=quick_mode,
            fast_eval=fast_eval
        )
        agent.train(total_timesteps=10000)  # Adjust timesteps as needed
        
        st.success("Training completed!")

if __name__ == "__main__":
    main()
