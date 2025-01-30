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
import numpy as np

from core.visualization import TradingVisualizer
from core.base_agent import UnifiedTradingAgent
from utils.stock_utils import parse_stock_list

def display_training_tab():
    """
    Renders the training interface tab with continuous action space support
    """
    st.header("Portfolio Allocation Agent Configuration")

    # Input parameters
    st.subheader("Training Options")
    stock_names = st.text_input("Training Stock Symbols (comma-separated)",
                               value="AAPL,MSFT,TSLA,GOOG,NVDA")
    st.session_state.stock_names = parse_stock_list(stock_names)

    # Environment parameters
    st.header("Environment Parameters")
    col1, col2 = st.columns(2)
    with col1:
        initial_balance = st.number_input("Initial Balance", value=10000)
    with col2:
        transaction_cost = st.number_input("Transaction Cost",
                                         value=0.001,
                                         step=0.0001,
                                         format="%.4f")

    st.session_state.env_params = {
        'initial_balance': initial_balance,
        'transaction_cost': transaction_cost,
        'use_position_profit': True,
        'use_holding_bonus': True,
        'use_trading_penalty': True
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
                         value=datetime.now() - timedelta(days=365)),
            datetime.min.time())

    st.session_state.train_start_date = train_start_date
    st.session_state.train_end_date = train_end_date

    tab1, tab2 = st.tabs(["Manual Parameters", "Hyperparameter Tuning"])

    with tab1:
        st.header("Policy Network Parameters")
        use_optuna_params = st.checkbox("Use Optuna Optimized Parameters",
                                      value=False)
        if not use_optuna_params:
            ppo_params = get_parameters(use_optuna_params)
            if st.button("Start Training"):
                run_training(ppo_params)
        else:
            if st.button("Start Training"):
                if st.session_state.ppo_params is None:
                    st.warning(
                        "Please run hyperparameter tuning before training model.")
                else:
                    run_training(st.session_state.ppo_params)

    with tab2:
        hyperparameter_tuning()

    if st.session_state.ppo_params is not None:
        display_testing_interface(st.session_state.ppo_params, use_optuna_params)

def get_parameters(use_optuna_params: bool) -> Dict[str, Any]:
    """Get PPO parameters for continuous action space."""
    if use_optuna_params and st.session_state.ppo_params is not None:
        st.info("Using Optuna's optimized parameters")
        return st.session_state.ppo_params

    col1, col2 = st.columns(2)
    with col1:
        learning_rate = st.number_input("Learning Rate",
                                      value=3e-4,
                                      format="%.1e")
        n_steps = st.number_input("Steps Per Update", value=2048)
        batch_size = st.number_input("Batch Size", value=64)
        n_epochs = st.number_input("Number of Epochs", value=10)

    with col2:
        gamma = st.number_input("Gamma (Discount Factor)", 
                              value=0.99,
                              format="%.3f")
        gae_lambda = st.number_input("GAE Lambda",
                                   value=0.95,
                                   format="%.3f")
        clip_range = st.number_input("PPO Clip Range",
                                   value=0.2,
                                   format="%.2f")

    return {
        'learning_rate': learning_rate,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_range': clip_range,
        'policy_kwargs': {
            'net_arch': [dict(pi=[128, 128], vf=[128, 128])]
        }
    }

def run_training(ppo_params: Dict[str, Any]) -> None:
    """Execute training with continuous action space support."""
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    callback = ProgressBarCallback(
        total_timesteps=(st.session_state.train_end_date -
                        st.session_state.train_start_date).days,
        progress_bar=progress_bar,
        status_placeholder=status_placeholder)

    metrics = st.session_state.model.train(
        stock_names=st.session_state.stock_names,
        start_date=st.session_state.train_start_date,
        end_date=st.session_state.train_end_date,
        env_params=st.session_state.env_params,
        ppo_params=ppo_params,
        callback=callback)

    if metrics:
        st.subheader("Training Parameters")
        param_cols = st.columns(3)
        for i, (param, value) in enumerate(ppo_params.items()):
            if param != 'policy_kwargs':
                with param_cols[i % 3]:
                    st.metric(param, f"{value:.2e}" if param == 'learning_rate' else f"{value}")

        display_training_metrics(metrics)

    st.session_state.ppo_params = ppo_params
    st.success("Training completed successfully!")

def display_training_metrics(metrics: Dict[str, float]) -> None:
    """Display training metrics with portfolio allocation focus."""
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

def hyperparameter_tuning() -> None:
    """Hyperparameter tuning interface for continuous action space."""
    st.header("Hyperparameter Tuning")

    with st.expander("Tuning Configuration", expanded=True):
        trials_number = st.number_input("Number of Trials",
                                      min_value=1,
                                      value=20,
                                      step=1)
        pruning_enabled = st.checkbox("Enable Pruning", value=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Network Parameters")
            lr_min = st.number_input("Learning Rate Min",
                                   value=1e-5,
                                   format="%.1e")
            lr_max = st.number_input("Learning Rate Max",
                                   value=1e-3,
                                   format="%.1e")

            steps_min = st.number_input("Steps Min",
                                      value=1024,
                                      step=256)
            steps_max = st.number_input("Steps Max",
                                      value=4096,
                                      step=256)

        with col2:
            st.subheader("PPO Parameters")
            gamma_min = st.number_input("Gamma Min",
                                      value=0.9,
                                      format="%.2f")
            gamma_max = st.number_input("Gamma Max",
                                      value=0.999,
                                      format="%.3f")

            gae_min = st.number_input("GAE Lambda Min",
                                    value=0.9,
                                    format="%.2f")
            gae_max = st.number_input("GAE Lambda Max",
                                    value=0.99,
                                    format="%.2f")

        optimization_metric = st.selectbox(
            "Optimization Target",
            ["sharpe_ratio", "sortino_ratio", "total_return"],
            help="Metric to optimize during hyperparameter search")

    if st.button("Start Hyperparameter Tuning"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner() if pruning_enabled else None)

        def objective(trial: optuna.Trial) -> float:
            ppo_params = {
                'learning_rate':
                trial.suggest_loguniform('learning_rate', lr_min, lr_max),
                'n_steps':
                trial.suggest_int('n_steps', steps_min, steps_max),
                'batch_size':
                trial.suggest_int('batch_size', 32, 256),
                'n_epochs':
                trial.suggest_int('n_epochs', 5, 15),
                'gamma':
                trial.suggest_uniform('gamma', gamma_min, gamma_max),
                'gae_lambda':
                trial.suggest_uniform('gae_lambda', gae_min, gae_max),
                'policy_kwargs': {
                    'net_arch': [dict(pi=[128, 128], vf=[128, 128])]
                }
            }

            status_text.text(
                f"Trial {trial.number + 1}/{trials_number}: Testing parameters")

            try:
                trial_model = UnifiedTradingAgent()
                metrics = trial_model.train(
                    stock_names=st.session_state.stock_names,
                    start_date=st.session_state.train_start_date,
                    end_date=st.session_state.train_end_date,
                    env_params=st.session_state.env_params,
                    ppo_params=ppo_params)

                trial_value = metrics.get(optimization_metric, float('-inf'))
                progress_bar.progress((trial.number + 1) / trials_number)
                return trial_value

            except Exception as e:
                st.error(f"Trial {trial.number} failed: {str(e)}")
                return float('-inf')

        try:
            study.optimize(objective, n_trials=trials_number)
            st.success("Hyperparameter optimization completed!")

            # Display results
            results_df = pd.DataFrame(
                [{
                    'Trial': t.number,
                    'Value': t.value,
                    **t.params
                } for t in study.trials if t.value is not None])

            tab1, tab2 = st.tabs(["Best Parameters", "Optimization History"])

            with tab1:
                st.subheader("Best Configuration")
                best_params = study.best_params
                best_value = study.best_value

                for param, value in best_params.items():
                    if param == 'learning_rate':
                        st.metric(param, f"{value:.2e}")
                    else:
                        st.metric(param, f"{value:.4f}")
                st.metric(f"Best {optimization_metric}", f"{best_value:.4f}")

                st.session_state.ppo_params = study.best_params

            with tab2:
                st.subheader("Optimization Progress")
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=results_df.index,
                              y=results_df['Value'],
                              mode='lines+markers',
                              name='Trial Value'))
                fig.update_layout(
                    title='Optimization History',
                    xaxis_title='Trial Number',
                    yaxis_title=optimization_metric.replace('_', ' ').title())
                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")

def display_testing_interface(ppo_params: Dict[str, Any],
                           use_optuna_params: bool = False) -> None:
    """Testing interface with portfolio weight visualization."""
    st.header("Model Testing")

    test_col1, test_col2 = st.columns(2)
    with test_col1:
        test_start_date = datetime.combine(
            st.date_input("Test Start Date",
                         value=datetime.now() - timedelta(days=180)),
            datetime.min.time())
    with test_col2:
        test_end_date = datetime.combine(
            st.date_input("Test End Date",
                         value=datetime.now()),
            datetime.min.time())

    if st.button("Test Model"):
        test_results = st.session_state.model.test(
            stock_names=st.session_state.stock_names,
            start_date=test_start_date,
            end_date=test_end_date,
            env_params=st.session_state.env_params)

        if test_results and 'metrics' in test_results:
            st.subheader("Test Results")

            # Portfolio metrics
            metrics_cols = st.columns(3)
            metrics = test_results['metrics']

            with metrics_cols[0]:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
            with metrics_cols[1]:
                st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
                st.metric("Volatility", f"{metrics['volatility']:.2%}")
            with metrics_cols[2]:
                st.metric("Information Ratio", f"{metrics['information_ratio']:.2f}")

            # Portfolio allocation visualization
            st.subheader("Portfolio Allocation Over Time")
            if 'portfolio_weights' in test_results:
                weights_fig = go.Figure()
                dates = [info['date'] for info in test_results['info_history']]
                weights = np.array([info['portfolio_weights'] for info in test_results['info_history']])

                for i, symbol in enumerate(st.session_state.stock_names):
                    weights_fig.add_trace(
                        go.Scatter(x=dates,
                                 y=weights[:, i],
                                 name=symbol,
                                 stackgroup='one'))

                weights_fig.update_layout(
                    title='Portfolio Weight Allocation',
                    yaxis_title='Weight',
                    xaxis_title='Date',
                    showlegend=True)
                st.plotly_chart(weights_fig)

            # Performance visualization
            if 'combined_plot' in test_results:
                st.plotly_chart(test_results['combined_plot'])