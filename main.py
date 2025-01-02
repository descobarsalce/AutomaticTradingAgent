
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
    
    # Training parameters
    st.header("Training Parameters")
    initial_balance = st.number_input("Initial Balance", value=10000)
    transaction_cost = st.number_input("Transaction Cost", value=0.0, step=0.001)
    min_transaction_size = st.number_input("Minimum Transaction Size", value=0.001, step=0.001)
    
    if st.button("Start Training"):
        # Load your data here
        data = pd.DataFrame()  # Replace with your data loading logic
        
        # Create environment with selected components
        env = SimpleTradingEnv(
            data=data,
            initial_balance=initial_balance,
            transaction_cost=transaction_cost,
            min_transaction_size=min_transaction_size,
            use_position_profit=use_position_profit,
            use_holding_bonus=use_holding_bonus,
            use_trading_penalty=use_trading_penalty
        )
        
        # Initialize and train agent
        agent = TradingAgent(env)
        agent.train(total_timesteps=10000)  # Adjust timesteps as needed
        
        st.success("Training completed!")

if __name__ == "__main__":
    main()
