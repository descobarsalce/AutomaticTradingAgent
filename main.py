import streamlit as st
import os
import logging
from utils.logging import setup_logging, display_logs
from utils.callbacks import ProgressBarCallback
from environment import SimpleTradingEnv
from core import TradingAgent
import pandas as pd

def init_session_state():
    """Initialize session state variables"""
    # Initialize logs list if not present
    if not hasattr(st.session_state, 'logs'):
        st.session_state.logs = []
    # Initialize PPO params if not present    
    if not hasattr(st.session_state, 'ppo_params'):
        st.session_state.ppo_params = None

def main():
    try:
        # Initialize session state first
        init_session_state()

        # Configure logging
        setup_logging(level='INFO')
        logger = logging.getLogger(__name__)

        # Log application start
        logger.info("Trading Agent Application Started")

        # Create sidebar for logs
        with st.sidebar:
            st.header("System Logs")
            log_container = st.container()
            display_logs(log_container)

        st.title("Trading Agent Configuration")
        logger.info("Loading configuration interface")

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
            learning_rate = st.number_input("Learning Rate", value=3e-4, format="%.1e")
            n_steps = st.number_input("Number of Steps", value=256)
            batch_size = st.number_input("Batch Size", value=128)
            n_epochs = st.number_input("Number of Epochs", value=3)
        with col4:
            gamma = st.number_input("Gamma (Discount Factor)", value=0.99)
            clip_range = st.number_input("Clip Range", value=0.2)
            target_kl = st.number_input("Target KL Divergence", value=0.05)

        # Add test buttons
        test_mode = st.checkbox("Test Mode (100 steps)", value=False)
        col_train, col_test = st.columns(2)

        if col_train.button("Start Training"):
            logger.info("Starting model training process")
            try:
                # Create progress tracking elements
                progress_bar = st.progress(0)
                status_placeholder = st.empty()

                # Create sample data
                logger.debug("Creating sample training data")
                data = pd.DataFrame({
                    'Open': [100] * 1000,
                    'High': [110] * 1000,
                    'Low': [90] * 1000,
                    'Close': [105] * 1000,
                    'Volume': [1000] * 1000
                })

                logger.info(f"Configuring environment with initial balance: {initial_balance}")
                env = SimpleTradingEnv(
                    data=data,
                    initial_balance=initial_balance,
                    transaction_cost=transaction_cost,
                    min_transaction_size=min_transaction_size,
                    max_position_pct=max_position_pct,
                    use_position_profit=use_position_profit,
                    use_holding_bonus=use_holding_bonus,
                    use_trading_penalty=use_trading_penalty,
                    training_mode=True
                )

                logger.info("Initializing agent parameters")
                st.session_state.ppo_params = {
                    'learning_rate': learning_rate,
                    'n_steps': n_steps,
                    'batch_size': batch_size,
                    'n_epochs': n_epochs,
                    'gamma': gamma,
                    'clip_range': clip_range,
                    'target_kl': target_kl
                }

                agent = TradingAgent(
                    env=env,
                    ppo_params=st.session_state.ppo_params,
                    quick_mode=quick_mode,
                    fast_eval=fast_eval
                )

                total_timesteps = 100 if test_mode else 10000
                logger.info(f"Starting training for {total_timesteps} timesteps")

                progress_callback = ProgressBarCallback(
                    total_timesteps=total_timesteps,
                    progress_bar=progress_bar,
                    status_placeholder=status_placeholder
                )

                agent.train(total_timesteps=total_timesteps, callback=progress_callback)
                agent.save("trained_model.zip")
                logger.info("Training completed successfully")
                st.success("Training completed and model saved!")

            except Exception as e:
                logger.error(f"Training failed: {str(e)}", exc_info=True)
                st.error(f"Error during training: {str(e)}")

        if col_test.button("Test Model"):
            try:
                model_path = "trained_model.zip"
                if not os.path.exists(model_path):
                    logger.warning("Model file not found")
                    st.error("Please train the model first before testing!")
                    return

                logger.info("Starting model testing")
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

                test_agent = TradingAgent(
                    env=test_env,
                    ppo_params=st.session_state.ppo_params,
                    quick_mode=quick_mode,
                    fast_eval=fast_eval
                )

                test_agent.load("trained_model.zip")

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

                        progress_bar.progress(steps / 100)
                        metrics_placeholder.write({
                            'Step': steps,
                            'Reward': round(total_reward, 2),
                            'Portfolio Value': round(info['net_worth'], 2),
                            'Position': round(float(test_env.shares_held), 3)
                        })

                    logger.info(f"Test completed with final portfolio value: ${info['net_worth']:.2f}")
                    st.success(f"Test completed! Final portfolio value: ${info['net_worth']:.2f}")

            except Exception as e:
                logger.error(f"Testing failed: {str(e)}", exc_info=True)
                st.error(f"Error during testing: {str(e)}")

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        if 'logger' in locals():
            logger.critical(f"Application failed: {str(e)}", exc_info=True)
        else:
            print(f"Startup error: {str(e)}")

if __name__ == "__main__":
    main()