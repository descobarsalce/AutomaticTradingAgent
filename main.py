import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from environment import SimpleTradingEnv
from core import TradingAgent
from data_handler import DataHandler
from visualization import TradingVisualizer
from callbacks import ProgressBarCallback

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
default_symbols = "AAPL,GOOGL"
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
initial_balance = st.sidebar.number_input("Initial Balance ($)", value=1000)

# Transaction parameters
transaction_cost = st.sidebar.number_input(
    "Transaction Cost (%)",
    min_value=0.0,
    max_value=5.0,
    value=0.001,
    step=0.001,
    help="Transaction cost as a percentage (e.g., 0.1% = 0.001)"
) / 100.0  # Convert percentage to decimal

min_transaction_size = st.sidebar.number_input(
    "Minimum Transaction Size ($)",
    min_value=0,
    max_value=10000,
    value=10,
    step=10,
    help="Minimum amount in dollars required for a trade"
)

training_steps = st.sidebar.number_input("Training Steps", value=10000, step=1000)
quick_mode = st.sidebar.checkbox("Enable Quick Training Mode", value=False, key="quick_mode")

# Hyperparameter optimization settings
st.sidebar.subheader("Hyperparameter Optimization")
enable_optimization = st.sidebar.checkbox("Enable Grid Search", value=False)
if enable_optimization:
    n_eval_episodes = st.sidebar.number_input("Evaluation Episodes per Combination", value=3, min_value=1, max_value=10)
    optimization_steps = st.sidebar.number_input("Steps per Combination", value=5000, step=1000)

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

from hyperparameter_optimizer import HyperparameterOptimizer

if train_model:
    st.session_state.environments = {}
    st.session_state.trained_agents = {}
    st.session_state.all_trades = {}
    st.session_state.optimization_results = {}
    
    progress_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    for idx, (symbol, data) in enumerate(st.session_state.portfolio_data.items()):
        progress = idx / len(st.session_state.portfolio_data)
        progress_placeholder.text(f"Processing {symbol}...")
        progress_bar.progress(progress)
        
        # Allocate initial balance based on weights
        symbol_balance = initial_balance * weights[symbol]
        st.session_state.environments[symbol] = SimpleTradingEnv(
            data=data,
            initial_balance=symbol_balance,
            transaction_cost=transaction_cost,
            min_transaction_size=min_transaction_size
        )
        
        if enable_optimization:
            # Initialize hyperparameter optimizer
            optimizer = HyperparameterOptimizer(st.session_state.environments[symbol])
            
            # Create progress indicators for optimization
            opt_progress = st.progress(0)
            opt_status = st.empty()
            
            with st.spinner(f"Optimizing hyperparameters for {symbol}..."):
                try:
                    best_params, results = optimizer.optimize(
                        total_timesteps=optimization_steps,
                        n_eval_episodes=n_eval_episodes,
                        progress_bar=opt_progress,
                        status_placeholder=opt_status
                    )
                    st.session_state.optimization_results[symbol] = optimizer.get_optimization_summary()
                    
                    # Initialize agent with best parameters
                    st.session_state.trained_agents[symbol] = TradingAgent(
                        st.session_state.environments[symbol],
                        ppo_params=best_params
                    )
                except Exception as e:
                    st.error(f"Error during hyperparameter optimization for {symbol}: {str(e)}")
                    continue
        else:
            # Initialize agent with default parameters and quick mode setting
            st.session_state.trained_agents[symbol] = TradingAgent(
                st.session_state.environments[symbol],
                quick_mode=st.session_state.get('quick_mode', False)
            )
        
        # Train agent with enhanced progress tracking
        training_progress = st.progress(0)
        training_status = st.empty()
        
        with st.spinner(f"Training agent for {symbol}..."):
            try:
                # Create callbacks for progress and metrics tracking
                progress_callback = ProgressBarCallback(
                    total_timesteps=training_steps,
                    progress_bar=training_progress,
                    status_placeholder=training_status
                )
                
                metrics_callback = PortfolioMetricsCallback(
                    eval_freq=max(100, training_steps // 20),  # Update metrics ~20 times during training
                    verbose=1
                )
                
                # Train with both callbacks
                st.session_state.trained_agents[symbol].train(
                    total_timesteps=training_steps,
                    callback=[progress_callback, metrics_callback]
                )
                
                # Display training metrics
                if metrics_callback.get_metrics():
                    st.subheader("Training Metrics")
                    metrics = metrics_callback.get_metrics()
                    
                    metrics_cols = st.columns(4)
                    metrics_cols[0].metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    metrics_cols[1].metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
                    metrics_cols[2].metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
                    metrics_cols[3].metric("Average Returns", f"{metrics['returns']:.2%}")
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
        
        # Display optimization results if available
        if enable_optimization and symbol in st.session_state.optimization_results:
            st.subheader(f"{symbol} Hyperparameter Optimization Results")
            opt_results = st.session_state.optimization_results[symbol]
            
            # Display optimization status with appropriate styling
            status = opt_results.get("status", "Unknown")
            if "Failed" in status:
                st.error(f"Optimization Status: {status}")
                st.error(opt_results.get("message", "An error occurred during optimization"))
            elif status == "Not started":
                st.warning("Optimization has not started yet")
            elif not opt_results.get("success", False):
                st.warning(opt_results.get("message", "Optimization did not complete successfully"))
            else:
                st.success(f"Optimization Status: {status}")
                st.success(opt_results.get("message", "Optimization completed successfully"))

            # Display parameters and results
            if opt_results.get("best_params"):
                st.write("Best Parameters Found:")
                st.json(opt_results["best_params"])
                if not opt_results.get("success", False):
                    st.info("Using default/fallback parameters due to optimization issues")
            
            if opt_results.get("top_5_results"):
                st.write("Top 5 Parameter Combinations:")
                results_df = pd.DataFrame(opt_results["top_5_results"])
                # Format the DataFrame for better display
                if not results_df.empty:
                    results_df['avg_reward'] = results_df['avg_reward'].round(4)
                    if 'sharpe_ratio' in results_df.columns:
                        results_df['sharpe_ratio'] = results_df['sharpe_ratio'].round(4)
                    st.dataframe(results_df)
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Total Combinations Tested", 
                    opt_results.get("total_combinations_tested", 0)
                )
            with col2:
                reward = opt_results.get("best_reward", 0)
                st.metric(
                    "Best Reward Achieved", 
                    f"{reward:.4f}" if isinstance(reward, (int, float)) else "N/A"
                )
        
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
