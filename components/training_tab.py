"""
Training Interface Component
Handles the model training and hyperparameter tuning interface
"""
import streamlit as st
from datetime import datetime, timedelta
import optuna
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, Optional
from utils.callbacks import ProgressBarCallback
from core.visualization import TradingVisualizer
import os


def hyperparameter_tuning(stock_name: str, train_start_date: datetime,
                          train_end_date: datetime,
                          env_params: Dict[str, Any]) -> None:
    """
    Interface for hyperparameter optimization using Optuna
    """
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

        def objective(trial: optuna.Trial) -> float:
            try:
                ppo_params = {
                    'learning_rate':
                    trial.suggest_loguniform('learning_rate', lr_min, lr_max),
                    'n_steps':
                    trial.suggest_int('n_steps', steps_min, steps_max),
                    'batch_size':
                    trial.suggest_int('batch_size', batch_min, batch_max),
                    'n_epochs':
                    trial.suggest_int('n_epochs', epochs_min, epochs_max),
                    'gamma':
                    trial.suggest_uniform('gamma', gamma_min, gamma_max),
                    'gae_lambda':
                    trial.suggest_uniform('gae_lambda', gae_min, gae_max),
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

                # Use selected optimization metric
                trial_value = metrics.get(optimization_metric, float('-inf'))
                progress = (trial.number + 1) / trials_number
                progress_bar.progress(progress)

                return trial_value

            except Exception as e:
                st.error(f"Error in trial {trial.number}: {str(e)}")
                return float('-inf')

        try:
            study.optimize(objective, n_trials=trials_number)
            st.success("Hyperparameter tuning completed!")

            # Save best parameters
            st.session_state.ppo_params = study.best_params

        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")


def display_training_tab():
    """
    Renders the training interface tab
    """
    st.header("Trading Agent Configuration")

    # Input parameters
    st.subheader("Training Options")
    stock_name = st.text_input("Training Stock Symbol", value="AAPL")

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

    # Training period selection
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
        ppo_params = display_manual_parameters()

    with tab2:
        hyperparameter_tuning(stock_name, train_start_date, train_end_date,
                              env_params)

    if st.button("Start Training"):
        run_training(stock_name, train_start_date, train_end_date, env_params,
                     ppo_params)

    display_testing_interface()


def display_manual_parameters() -> Dict[str, Any]:
    """
    Displays and handles manual parameter input interface
    Returns:
        Dictionary of PPO parameters
    """
    st.header("Agent Parameters")

    use_optuna_params = st.checkbox("Use Optuna Optimized Parameters",
                                    value=False)

    if use_optuna_params and st.session_state.ppo_params is not None:
        st.info("Using Optuna's optimized parameters")
        params = st.session_state.ppo_params

        col3, col4 = st.columns(2)
        with col3:
            st.text(f"Learning Rate: {params['learning_rate']:.2e}")
            st.text(f"PPO Steps: {params['n_steps']}")
            st.text(f"Batch Size: {params['batch_size']}")
            st.text(f"Number of Epochs: {params['n_epochs']}")
        with col4:
            st.text(f"Gamma: {params['gamma']:.4f}")
            st.text(f"GAE Lambda: {params['gae_lambda']:.4f}")

        return {
            **params,
            'clip_range': 0.2,  # Default value for non-tuned parameter
            'target_kl': 0.05  # Default value for non-tuned parameter
        }
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

        return {
            'learning_rate': learning_rate,
            'n_steps': ppo_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'clip_range': clip_range,
            'target_kl': target_kl
        }


def run_training(stock_name: str, train_start_date: datetime,
                 train_end_date: datetime, env_params: Dict[str, Any],
                 ppo_params: Dict[str, Any]) -> None:
    """
    Executes the training process and displays results
    """
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    progress_callback = ProgressBarCallback(
        total_timesteps=(train_end_date - train_start_date).days,
        progress_bar=progress_bar,
        status_placeholder=status_placeholder)

    metrics = st.session_state.model.stock_name = stock_name  # Store stock name
    st.session_state.model.train(stock_name=stock_name,
                                           start_date=train_start_date,
                                           end_date=train_end_date,
                                           env_params=env_params,
                                           ppo_params=ppo_params,
                                           callback=progress_callback)

    if metrics:
        display_training_metrics(metrics)

    st.success("Training completed and model saved!")


def display_training_metrics(metrics: Dict[str, float]) -> None:
    """
    Displays the training metrics in a formatted layout
    """
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        st.metric("Maximum Drawdown", f"{metrics['max_drawdown']:.2%}")
    with metrics_col2:
        st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
        st.metric("Volatility", f"{metrics['volatility']:.2%}")
    with metrics_col3:
        st.metric("Total Return", f"{metrics['total_return']:.2%}")
        st.metric("Final Portfolio Value", f"${metrics['final_value']:,.2f}")


def display_testing_interface() -> None:
    """
    Displays the testing interface and visualization options
    """
    st.header("Testing Interface")
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

    if st.button("Test Model"):
        if not os.path.exists("trained_model.zip"):
            st.error("No trained model found. Please train a model first.")
        else:
            test_results = st.session_state.model.test(
                stock_name=st.session_state.model.stock_name,
                start_date=test_start_date,
                end_date=test_end_date,
                env_params=env_params,
                ppo_params=ppo_params)

            # Display test metrics
            if test_results and 'metrics' in test_results:
                metrics = test_results['metrics']

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
                with col2:
                    st.metric("Sortino Ratio",
                              f"{metrics['sortino_ratio']:.2f}")
                    st.metric("Volatility", f"{metrics['volatility']:.2%}")
                with col3:
                    if 'information_ratio' in metrics:
                        st.metric("Information Ratio",
                                  f"{metrics['information_ratio']:.2f}")

                # Display performance charts
                if 'combined_plot' in test_results:
                    st.plotly_chart(test_results['combined_plot'])

    st.header("Visualization Options")
    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        show_rsi = st.checkbox("Show RSI", value=True, key="training_rsi")
        show_sma20 = st.checkbox("Show SMA 20",
                                 value=True,
                                 key="training_sma20")

    with plot_col2:
        show_sma50 = st.checkbox("Show SMA 50",
                                 value=True,
                                 key="training_sma50")
        rsi_period = st.slider("RSI Period",
                               min_value=7,
                               max_value=21,
                               value=14,
                               key="training_rsi_period") if show_rsi else 14

    if st.button("Generate Charts"):
        generate_test_charts(test_start_date, test_end_date, show_rsi,
                             show_sma20, show_sma50, rsi_period)


def generate_test_charts(test_start_date: datetime, test_end_date: datetime,
                         show_rsi: bool, show_sma20: bool, show_sma50: bool,
                         rsi_period: int) -> None:
    """
    Generates and displays test charts
    """
    with st.spinner("Fetching and processing data..."):
        try:
            portfolio_data = st.session_state.model.data_handler.fetch_data(
                st.session_state.model.stock_name, test_start_date,
                test_end_date)

            if not portfolio_data:
                st.error(
                    "No data available for the selected symbol and date range."
                )
            else:
                portfolio_data = st.session_state.model.data_handler.prepare_data(
                )

                if st.session_state.model.stock_name in portfolio_data:
                    data = portfolio_data[st.session_state.model.stock_name]

                    visualizer = TradingVisualizer()
                    visualizer.show_rsi = show_rsi
                    visualizer.show_sma20 = show_sma20
                    visualizer.show_sma50 = show_sma50
                    visualizer.rsi_period = rsi_period

                    st.subheader("Technical Analysis")
                    main_chart = visualizer.create_single_chart(
                        st.session_state.model.stock_name, data)
                    if main_chart:
                        st.plotly_chart(main_chart, use_container_width=True)

        except Exception as e:
            st.error(f"Error generating charts: {str(e)}")
