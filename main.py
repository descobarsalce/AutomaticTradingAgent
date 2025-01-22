"""
Trading Agent Web Interface
A Streamlit-based dashboard for configuring and managing reinforcement learning trading agents.

System Architecture:
- Web Interface Layer (Streamlit): Provides UI for configuration and visualization
- Agent Layer (UnifiedTradingAgent): Implements trading strategy and training logic
- Environment Layer (SimpleTradingEnv): Simulates market interactions
- Data Layer: Handles market data acquisition and preprocessing

Key Components:
1. Configuration Interface:
   - Agent hyperparameter tuning
   - Environment settings adjustment
   - Training period selection

2. Training Pipeline:
   - Data preparation
   - Model training with progress tracking
   - Performance metrics calculation

3. Testing Interface:
   - Model evaluation on unseen data
   - Performance visualization
   - Trading behavior analysis

4. Monitoring System:
   - Real-time training progress
   - Portfolio value tracking
   - Transaction logging

Usage Example:
```python
# Initialize the application
streamlit run TradingAgentUI.py

# Configure agent parameters in UI:
# - Set stock symbol (e.g., 'AAPL')
# - Adjust initial balance
# - Tune learning parameters
# - Select date ranges

# Train the agent:
# Click "Start Training" to begin the training process
# Monitor progress in the sidebar logs

# Test the agent:
# Click "Test Model" to evaluate on new data
# View performance metrics and visualizations
```
"""

import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from utils.callbacks import ProgressBarCallback
from core.base_agent import UnifiedTradingAgent
from core.visualization import TradingVisualizer
import optuna  # Added import for Optuna

# Configure logging
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def init_session_state() -> None:
    """
    Initialize Streamlit session state variables for persistent storage across reruns.

    Initializes:
        - log_messages: List[str] - Chronological log messages
        - ppo_params: Dict[str, Union[float, int, bool]] - PPO algorithm configuration
        - model: UnifiedTradingAgent - Trading agent model instance

    Implementation:
        The function checks for each required key in st.session_state and
        initializes it if missing. This ensures persistence across Streamlit reruns
        while avoiding reinitializing existing state.

    Example:
        ```python
        # Initialize state at app startup
        init_session_state()

        # Access state variables
        model = st.session_state.model
        logs = st.session_state.log_messages
        ```
    """
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    if 'ppo_params' not in st.session_state:
        st.session_state.ppo_params = None
    if 'model' not in st.session_state:
        st.session_state.model = UnifiedTradingAgent()


class StreamlitLogHandler(logging.Handler):
    """
    Custom logging handler that redirects log messages to Streamlit's interface.

    Maintains a fixed-size buffer of recent log messages in the session state.
    Implements the observer pattern to capture logs from all components.

    Attributes:
        MAX_LOGS: int - Maximum number of logs to maintain in memory (100)
        format: Callable - Log message formatter function

    Implementation Details:
        - Uses Streamlit session state for persistence
        - Implements circular buffer behavior for log storage
        - Thread-safe for concurrent logging

    Example:
        ```python
        # Setup logging
        handler = StreamlitLogHandler()
        logger.addHandler(handler)

        # Log messages will appear in Streamlit sidebar
        logger.info("Training started...")
        ```
    """

    MAX_LOGS: int = 100

    def emit(self, record: logging.LogRecord) -> None:
        """
        Process a log record by formatting it and adding to the session state.

        Args:
            record (logging.LogRecord): The log record containing:
                - msg: str - The log message
                - levelno: int - Logging level number
                - created: float - Time when the log was created
                - args: tuple - Message format arguments

        Implementation:
            1. Formats the log record using the handler's formatter
            2. Appends to session state log buffer
            3. Maintains maximum log count by removing oldest entries
            4. Handles formatting exceptions gracefully

        Raises:
            Exception: If logging fails, error is printed but not propagated
        """
        try:
            log_entry: str = self.format(record)
            if 'log_messages' in st.session_state:
                st.session_state.log_messages.append(log_entry)
                if len(st.session_state.log_messages) > self.MAX_LOGS:
                    st.session_state.log_messages = st.session_state.log_messages[
                        -self.MAX_LOGS:]
            print(log_entry)  # Backup output
        except Exception as e:
            print(f"Logging error: {e}")


def hyperparameter_tuning(stock_name: str, train_start_date: datetime,
                          train_end_date: datetime,
                          env_params: Dict[str, Any]) -> None:
    st.header("Hyperparameter Tuning Options")

    with st.expander("Tuning Configuration", expanded=True):
        trials_number = st.number_input("Number of Trials",
                                        min_value=1,
                                        value=20,
                                        step=1)
        pruning_enabled = st.checkbox("Enable Early Trial Pruning", value=True)

        st.subheader("Parameter Search Ranges")

        col1, col2 = st.columns(2)
        with col1:
            lr_min = st.number_input("Learning Rate Min",
                                     value=1e-5,
                                     format="%.1e")
            lr_max = st.number_input("Learning Rate Max",
                                     value=5e-4,
                                     format="%.1e")
            steps_min = st.number_input("Steps Min", value=512, step=64)
            steps_max = st.number_input("Steps Max", value=2048, step=64)
            batch_min = st.number_input("Batch Size Min", value=64, step=32)
            batch_max = st.number_input("Batch Size Max", value=512, step=32)

        with col2:
            epochs_min = st.number_input("Training Epochs Min",
                                         value=3,
                                         step=1)
            epochs_max = st.number_input("Training Epochs Max",
                                         value=10,
                                         step=1)
            gamma_min = st.number_input("Gamma Min",
                                        value=0.90,
                                        step=0.01,
                                        format="%.3f")
            gamma_max = st.number_input("Gamma Max",
                                        value=0.999,
                                        step=0.001,
                                        format="%.3f")
            gae_min = st.number_input("GAE Lambda Min",
                                      value=0.90,
                                      step=0.01,
                                      format="%.2f")
            gae_max = st.number_input("GAE Lambda Max",
                                      value=0.99,
                                      step=0.01,
                                      format="%.2f")

        optimization_metric = st.selectbox(
            "Optimization Metric",
            ["sharpe_ratio", "sortino_ratio", "total_return"],
            help="Metric to optimize during hyperparameter search")

    if st.button("Start Hyperparameter Tuning"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create study with pruning
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner() if pruning_enabled else None)
        status_text = st.empty()

        study = optuna.create_study(direction='maximize')

        def objective(trial: optuna.Trial) -> float:
            try:
                ppo_params = {
                    'learning_rate':
                    trial.suggest_loguniform('learning_rate', 1e-5, 5e-4),
                    'n_steps':
                    trial.suggest_int('n_steps', 512, 2048),
                    'batch_size':
                    trial.suggest_int('batch_size', 64, 512),
                    'n_epochs':
                    trial.suggest_int('n_epochs', 3, 10),
                    'gamma':
                    trial.suggest_uniform('gamma', 0.90, 0.999),
                    'gae_lambda':
                    trial.suggest_uniform('gae_lambda', 0.90, 0.99),
                }

                status_text.text(
                    f"Trial {trial.number + 1}/{trials_number}: Testing parameters {ppo_params}"
                )

                # Train with current parameters
                metrics = st.session_state.model.train(
                    stock_name=stock_name,
                    start_date=train_start_date,
                    end_date=train_end_date,
                    env_params=env_params,
                    ppo_params=ppo_params)

                # Use Sharpe ratio as optimization metric
                trial_value = metrics.get('sharpe_ratio', float('-inf'))
                progress = (trial.number + 1) / trials_number
                progress_bar.progress(progress)

                return trial_value

            except Exception as e:
                st.error(f"Error in trial {trial.number}: {str(e)}")
                return float('-inf')

        try:
            study.optimize(objective, n_trials=trials_number)

            st.success("Hyperparameter tuning completed!")

            # Create detailed results dataframe
            trials_df = pd.DataFrame([{
                'Trial': t.number,
                'Value': t.value,
                **t.params
            } for t in study.trials if t.value is not None])

            # Display results in tabs
            tab1, tab2, tab3 = st.tabs([
                "Best Parameters", "Optimization History",
                "Parameter Importance"
            ])

            with tab1:
                st.subheader("Best Configuration Found")
                for param, value in study.best_params.items():
                    if param == 'learning_rate':
                        st.metric(f"Best {param}", f"{value:.2e}")
                    elif param in ['gamma', 'gae_lambda']:
                        st.metric(f"Best {param}", f"{value:.4f}")
                    else:
                        st.metric(f"Best {param}", f"{int(value)}")
                st.metric(f"Best {optimization_metric}",
                          f"{study.best_value:.6f}")

            with tab2:
                st.subheader("Trial History")
                history_fig = go.Figure()
                history_fig.add_trace(
                    go.Scatter(x=trials_df.index,
                               y=trials_df['Value'],
                               mode='lines+markers',
                               name='Trial Value'))
                history_fig.update_layout(
                    title='Optimization History',
                    xaxis_title='Trial Number',
                    yaxis_title=optimization_metric.replace('_', ' ').title())
                st.plotly_chart(history_fig)

            with tab3:
                st.subheader("Parameter Importance")
                importance_dict = optuna.importance.get_param_importances(
                    study)
                importance_df = pd.DataFrame({
                    'Parameter':
                    list(importance_dict.keys()),
                    'Importance':
                    list(importance_dict.values())
                }).sort_values('Importance', ascending=True)

                importance_fig = go.Figure()
                importance_fig.add_trace(
                    go.Bar(x=importance_df['Importance'],
                           y=importance_df['Parameter'],
                           orientation='h'))
                importance_fig.update_layout(
                    title='Parameter Importance Analysis',
                    xaxis_title='Relative Importance',
                    yaxis_title='Parameter',
                    height=400)
                st.plotly_chart(importance_fig)

            # Save full results to CSV
            trials_df.to_csv('hyperparameter_tuning_results.csv', index=False)

            # Download button for results
            st.download_button("Download Complete Results CSV",
                               trials_df.to_csv(index=False),
                               "hyperparameter_tuning_results.csv", "text/csv")

            # Save best parameters
            st.session_state.ppo_params = study.best_params

        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            logger.exception("Hyperparameter optimization error")


def main() -> None:
    init_session_state()

    st.title("Trading Agent Configuration")

    # Input parameters
    st.header("Test Options")
    stock_name = st.text_input("Stock Name", value="AAPL")

    # Environment parameters
    st.header("Environment Parameters")
    col1, col2 = st.columns(2)
    with col1:
        initial_balance = st.number_input("Initial Balance", value=10000)

    with col2:
        transaction_cost = st.number_input("Transaction Cost",
                                           value=0.01,
                                           step=0.001)

    env_params = {
        'initial_balance': initial_balance,
        'transaction_cost': transaction_cost,
        'use_position_profit': False,
        'use_holding_bonus': False,
        'use_trading_penalty': False
    }

    # Date selection for training
    st.subheader("Training Period")
    train_col1, train_col2 = st.columns(2)
    with train_col1:
        train_start_date = datetime.combine(
            st.date_input("Training Start Date",
                          value=datetime.now() - timedelta(days=365 * 5)),
            datetime.min.time())
    with train_col2:
        train_end_date = datetime.combine(
            st.date_input("Training End Date",
                          value=datetime.now() - timedelta(days=365 + 1)),
            datetime.min.time())

    tab1, tab2 = st.tabs(["Manual Parameters", "Hyperparameter Tuning"])

    with tab1:
        # Original manual parameter selection
        st.header("Agent Parameters")

        # Add checkbox to use Optuna parameters
        use_optuna_params = st.checkbox("Use Optuna Optimized Parameters",
                                        value=False)

        if use_optuna_params and st.session_state.ppo_params is not None:
            st.info("Using Optuna's optimized parameters")
            # Display Optuna parameters as read-only
            col3, col4 = st.columns(2)
            with col3:
                st.text(
                    f"Learning Rate: {st.session_state.ppo_params['learning_rate']:.2e}"
                )
                st.text(f"PPO Steps: {st.session_state.ppo_params['n_steps']}")
                st.text(
                    f"Batch Size: {st.session_state.ppo_params['batch_size']}")
                st.text(
                    f"Number of Epochs: {st.session_state.ppo_params['n_epochs']}"
                )
            with col4:
                st.text(f"Gamma: {st.session_state.ppo_params['gamma']:.4f}")
                st.text(
                    f"GAE Lambda: {st.session_state.ppo_params['gae_lambda']:.4f}"
                )

            # Store Optuna values in variables
            learning_rate = st.session_state.ppo_params['learning_rate']
            ppo_steps = st.session_state.ppo_params['n_steps']
            batch_size = st.session_state.ppo_params['batch_size']
            n_epochs = st.session_state.ppo_params['n_epochs']
            gamma = st.session_state.ppo_params['gamma']
            gae_lambda = st.session_state.ppo_params['gae_lambda']
            clip_range = 0.2  # Default value for non-tuned parameter
            target_kl = 0.05  # Default value for non-tuned parameter
        else:
            if use_optuna_params:
                st.warning(
                    "No Optuna parameters available. Please run hyperparameter tuning first."
                )

            col3, col4 = st.columns(2)
            with col3:
                learning_rate = st.number_input("Learning Rate",
                                                value=3e-4,
                                                format="%.1e")
                ppo_steps = st.number_input("PPO Steps Per Update", value=512)
                batch_size = st.number_input("Batch Size", value=128)
                n_epochs = st.number_input("Number of Epochs", value=5)
            with col4:
                gamma = st.number_input("Gamma (Discount Factor)", value=0.99)
                clip_range = st.number_input("Clip Range", value=0.2)
                target_kl = st.number_input("Target KL Divergence", value=0.05)

    with tab2:
        # Hyperparameter tuning section
        hyperparameter_tuning(stock_name, train_start_date, train_end_date,
                              env_params)

    col_train, col_test = st.columns(2)

    if col_train.button("Start Training"):
        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        ppo_params = {
            'learning_rate': learning_rate,
            'n_steps': ppo_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'clip_range': clip_range,
            'target_kl': target_kl
        }

        progress_callback = ProgressBarCallback(
            total_timesteps=(train_end_date - train_start_date).days,
            progress_bar=progress_bar,
            status_placeholder=status_placeholder)

        metrics = st.session_state.model.train(stock_name=stock_name,
                                               start_date=train_start_date,
                                               end_date=train_end_date,
                                               env_params=env_params,
                                               ppo_params=ppo_params,
                                               callback=progress_callback)

        if metrics:
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                st.metric("Maximum Drawdown", f"{metrics['max_drawdown']:.2%}")
            with metrics_col2:
                st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
                st.metric("Volatility", f"{metrics['volatility']:.2%}")
            with metrics_col3:
                st.metric("Total Return", f"{metrics['total_return']:.2%}")
                st.metric("Final Portfolio Value",
                          f"${metrics['final_value']:,.2f}")

        st.success("Training completed and model saved!")

    # Test period dates
    st.subheader("Test Period")
    test_col1, test_col2 = st.columns(2)
    with test_col1:
        test_start_date = datetime.combine(
            st.date_input("Test Start Date",
                          value=datetime.now() - timedelta(days=365)),
            datetime.min.time())
    with test_col2:
        test_end_date = datetime.combine(
            st.date_input("Test End Date", value=datetime.now()),
            datetime.min.time())

    # Plot controls
    st.header("Visualization Options")
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        show_rsi = st.checkbox("Show RSI", value=True)
        show_sma20 = st.checkbox("Show SMA 20", value=True)

    with plot_col2:
        show_sma50 = st.checkbox("Show SMA 50", value=True)
        rsi_period = st.slider(
            "RSI Period", min_value=7, max_value=21,
            value=14) if show_rsi else 14

    if st.button("Generate Charts"):
        with st.spinner("Fetching and processing data..."):
            try:
                portfolio_data = st.session_state.model.data_handler.fetch_data(
                    stock_name, test_start_date, test_end_date)
                if not portfolio_data:
                    st.error(
                        "No data available for the selected symbol and date range."
                    )
                else:
                    portfolio_data = st.session_state.model.data_handler.prepare_data()

                    if stock_name in portfolio_data:
                        data = portfolio_data[stock_name]

                        # Create TradingVisualizer instance with user preferences
                        visualizer = TradingVisualizer()
                        visualizer.show_rsi = show_rsi
                        visualizer.show_sma20 = show_sma20
                        visualizer.show_sma50 = show_sma50
                        visualizer.rsi_period = rsi_period

                        # Technical Analysis Charts
                        st.subheader("Technical Analysis")
                        main_chart = visualizer.create_single_chart(
                            stock_name, data)
                        if main_chart:
                            st.plotly_chart(main_chart,
                                            use_container_width=True)

                        # Create two columns for charts
                        col1, col2 = st.columns(2)

                        with col1:
                            # Plot cumulative returns
                            cum_returns_fig = visualizer.plot_cumulative_returns(
                                {stock_name: data})
                            st.plotly_chart(cum_returns_fig,
                                            use_container_width=True)

                            # Plot drawdown
                            drawdown_fig = visualizer.plot_drawdown(
                                {stock_name: data}, stock_name)
                            st.plotly_chart(drawdown_fig,
                                            use_container_width=True)

                        with col2:
                            # Plot performance and drawdown combined
                            perf_dd_fig = visualizer.plot_performance_and_drawdown(
                                {stock_name: data}, stock_name)
                            st.plotly_chart(perf_dd_fig,
                                            use_container_width=True)

                        # Metrics Display
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(
                            3)

                        try:
                            latest_close = data['Close'].iloc[-1]
                            price_change = (latest_close -
                                            data['Close'].iloc[0]
                                            ) / data['Close'].iloc[0] * 100

                            metrics_col1.metric("Latest Close",
                                                f"${latest_close:.2f}",
                                                f"{price_change:.2f}%")

                            avg_volume = data['Volume'].mean()
                            volume_change = (data['Volume'].iloc[-1] -
                                             avg_volume) / avg_volume * 100

                            metrics_col2.metric("Average Volume",
                                                f"{avg_volume:,.0f}",
                                                f"{volume_change:.2f}%")

                            if show_rsi and 'RSI' in data.columns:
                                latest_rsi = data['RSI'].iloc[-1]
                                rsi_change = latest_rsi - data['RSI'].iloc[
                                    -2] if len(data) > 1 else 0

                                metrics_col3.metric("Current RSI",
                                                    f"{latest_rsi:.2f}",
                                                    f"{rsi_change:.2f}")

                        except Exception as e:
                            st.error(f"Error calculating metrics: {str(e)}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    if col_test.button("Test Model"):
        try:
            if not os.path.exists("trained_model.zip"):
                st.error("Please train the model first before testing!")
                return

            env_params = {
                'initial_balance': initial_balance,
                'transaction_cost': transaction_cost,
            }

            # Use either Optuna optimized or manual parameters
            test_ppo_params = st.session_state.ppo_params if use_optuna_params else {
                'learning_rate': learning_rate,
                'n_steps': ppo_steps,
                'batch_size': batch_size,
                'n_epochs': n_epochs,
                'gamma': gamma,
                'clip_range': clip_range,
                'target_kl': target_kl
            }

            test_results = st.session_state.model.test(
                stock_name=stock_name,
                start_date=test_start_date,
                end_date=test_end_date,
                env_params=env_params,
                ppo_params=test_ppo_params)

            with st.expander("Test Results", expanded=True):
                progress_bar = st.progress(0)
                metrics_placeholder = st.empty()

                col1, col2, col3 = st.columns(3)
                metrics = test_results['metrics']

                with col1:
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    st.metric("Maximum Drawdown",
                              f"{metrics['max_drawdown']:.2%}")
                    st.metric(
                        "Total Return",
                        f"{(test_results['portfolio_history'][-1] - test_results['portfolio_history'][0]) / test_results['portfolio_history'][0]:.2%}"
                    )

                with col2:
                    st.metric("Sortino Ratio",
                              f"{metrics['sortino_ratio']:.2f}")
                    st.metric("Information Ratio",
                              f"{metrics['information_ratio']:.2f}")
                    st.metric("Volatility", f"{metrics['volatility']:.2%}")

                with col3:
                    st.metric(
                        "Final Portfolio Value",
                        f"${test_results['portfolio_history'][-1]:,.2f}")
                    st.metric("Initial Balance",
                              f"${test_results['portfolio_history'][0]:,.2f}")

                # Plot portfolio value over time
                st.subheader("Portfolio Value Over Time")
                st.line_chart(
                    pd.DataFrame(test_results['portfolio_history'],
                                 columns=['Portfolio Value']))

                # Create columns for charts
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    if len(test_results['returns']) > 0:
                        fig = go.Figure(data=[
                            go.Histogram(x=test_results['returns'], nbinsx=50)
                        ])
                        fig.update_layout(title="Returns Distribution",
                                          xaxis_title="Return",
                                          yaxis_title="Frequency",
                                          showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)

                    values = np.array(test_results['portfolio_history'])
                    peak = np.maximum.accumulate(values)
                    drawdowns = (peak - values) / peak
                    st.subheader("Drawdown Over Time")
                    st.area_chart(pd.DataFrame(drawdowns,
                                               columns=['Drawdown']))

                with chart_col2:
                    st.subheader("Agent Actions")
                    st.plotly_chart(test_results['action_plot'],
                                    use_container_width=True)

                    st.subheader("Price and Actions")
                    st.plotly_chart(test_results['combined_plot'],
                                    use_container_width=True)

                    st.subheader("Cumulative Returns")
                    cum_returns = pd.DataFrame(
                        np.cumprod(1 + test_results['returns']) - 1,
                        columns=['Returns'])
                    st.line_chart(cum_returns)

                    st.subheader("30-Day Rolling Volatility")
                    rolling_vol = pd.DataFrame(
                        test_results['returns'],
                        columns=['Returns']).rolling(30).std() * np.sqrt(252)
                    st.line_chart(rolling_vol)

        except Exception as e:
            st.error(f"Error during testing: {str(e)}")

    # Configure logging
    handler = StreamlitLogHandler()
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    # Create sidebar for logs
    with st.sidebar:
        st.header("Logs")
        for log in st.session_state.log_messages:
            st.text(log)


if __name__ == "__main__":
    main()
