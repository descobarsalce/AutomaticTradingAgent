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

import matplotlib.pyplot as plt

from utils.callbacks import ProgressBarCallback

from core.visualization import TradingVisualizer

from core.base_agent import UnifiedTradingAgent
import os
import numpy as np
from utils.stock_utils import parse_stock_list


def display_training_tab():
    """
    Renders the training interface tab
    """
    st.header("Trading Agent Configuration")

    # Input parameters
    st.subheader("Training Options")
    stock_names = st.text_input("Training Stock Symbol",
                                value="AAPL,MSFT,TSLA,GOOG,NVDA")
    st.session_state.stock_names = parse_stock_list(stock_names)

    # Environment parameters
    st.header("Environment Parameters")
    col1, col2 = st.columns(2)
    with col1:
        initial_balance = st.number_input("Initial Balance", value=10000)

    with col2:
        transaction_cost = st.number_input("Transaction Cost",
                                           value=0.01,
                                           step=0.001)

    st.session_state.env_params = {
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

    st.session_state.train_start_date = train_start_date
    st.session_state.train_end_date = train_end_date

    tab1, tab2 = st.tabs(["Manual Parameters", "Hyperparameter Tuning"])

    with tab1:
        st.header("Agent Parameters")
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
                        "Please run hyperparameter tuning before training model."
                    )
                else:
                    # Note that this will only work in the optimizaiton has already been run so that it has
                    run_training(st.session_state.ppo_params)

    with tab2:
        hyperparameter_tuning()

    if st.session_state.ppo_params is not None:
        display_testing_interface(st.session_state.ppo_params,
                                  use_optuna_params)

    # Add Python code execution interface
    st.header("Data Analysis Console")
    with st.expander("Python Code Execution", expanded=True):
        code = st.text_area(
            "Enter Python code:",
            height=300,
            help="Access data via st.session_state.model.data_handler")

        # Initialize persistent namespace in session state if not exists
        if 'code_namespace' not in st.session_state:
            st.session_state.code_namespace = {
                'np': np,
                'pd': pd,
                'plt': plt,
                'go': go,
                'vars': {},  # For user-defined variables
            }

        if st.button("Execute Code"):
            try:
                # Update namespace with latest session state
                st.session_state.code_namespace.update({
                    'data_handler':
                    st.session_state.model.data_handler,
                    'stock_names':
                    st.session_state.stock_names,
                    'train_start_date':
                    st.session_state.train_start_date,
                    'train_end_date':
                    st.session_state.train_end_date,
                    'test_start_date':
                    st.session_state.test_start_date,
                    'test_end_date':
                    st.session_state.test_end_date,
                    'env_params':
                    st.session_state.env_params,
                    'model':
                    st.session_state.model,
                    'vars':
                    st.session_state.
                    code_namespace['vars'],  # Preserve user variables
                })

                # Create reference to vars dict for easier access
                locals().update(st.session_state.code_namespace['vars'])

                # Create string buffer to capture print output
                import io
                import sys
                output_buffer = io.StringIO()
                original_stdout = sys.stdout
                sys.stdout = output_buffer

                # Execute the code and capture output
                with st.spinner("Executing code..."):
                    exec(code, globals(), st.session_state.code_namespace)

                    # Save all newly defined variables
                    st.session_state.code_namespace['vars'].update({
                        k: v
                        for k, v in st.session_state.code_namespace.items()
                        if k not in [
                            'np', 'pd', 'plt', 'go', 'data_handler',
                            'stock_names', 'train_start_date',
                            'train_end_date', 'test_start_date',
                            'test_end_date', 'env_params', 'model', 'vars'
                        ]
                    })

                    # Display any generated plots
                    if 'plt' in st.session_state.code_namespace:
                        st.pyplot(plt.gcf())
                        plt.close()

                    # Get and display the captured output
                    sys.stdout = original_stdout
                    output = output_buffer.getvalue()
                    if output:
                        st.text_area("Output:", value=output, height=250)

            except Exception as e:
                st.error(f"Error executing code: {str(e)}")
            finally:
                # Ensure stdout is restored
                sys.stdout = original_stdout


def get_parameters(use_optuna_params) -> Dict[str, Any]:
    """
    Displays and handles manual parameter input interface
    Returns:
        Dictionary of PPO parameters
    """

    if use_optuna_params:
        if st.session_state.ppo_params is not None:
            # If the code has already found optimal parameters (stored in the state session)
            st.info("Using Optuna's optimized parameters")
            params = st.session_state.ppo_params
            return {}, use_optuna_params
    elif use_optuna_params:
        if st.session_state.ppo_params is None:
            st.warning(
                "No Optuna parameters available. Please run hyperparameter tuning first."
            )
    else:
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


def run_training(ppo_params: Dict[str, Any]) -> None:
    """
    Executes the training process and displays results
    """
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    trade_history = []

    progress_callback = ProgressBarCallback(
        total_timesteps=(st.session_state.train_start_date -
                         st.session_state.train_end_date).days,
        progress_bar=progress_bar,
        status_placeholder=status_placeholder)

    metrics = st.session_state.model.train(
        stock_names=st.session_state.stock_names,
        start_date=st.session_state.train_start_date,
        end_date=st.session_state.train_end_date,
        env_params=st.session_state.env_params,
        ppo_params=ppo_params,
        callback=progress_callback)

    if metrics:
        # Display parameters used for testing, automatically sorting into columns:
        st.subheader("Parameters Used for Training")
        col1, col2, col3 = st.columns(3)
        index_col = 0
        all_cols = [col1, col2, col3]
        for param, value in ppo_params.items():
            with all_cols[index_col % 3]:
                st.metric(param, value)
                index_col += 1

        display_training_metrics(metrics)

    # Display trade history using TradingVisualizer
    if hasattr(st.session_state.model.env, '_trade_history'):
        TradingVisualizer.display_trade_history(
            st.session_state.model.env._trade_history, "Training History",
            "training_trade")

        # Option to download trade history
        st.download_button("Download Trade History",
                           trade_df.to_csv(index=False), "trade_history.csv",
                           "text/csv")

    st.session_state.ppo_params = ppo_params
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


def hyperparameter_tuning() -> None:
    """
    Interface for hyperparameter optimization using Optuna
    """

    stock_names = st.session_state.stock_names
    train_start_date = st.session_state.train_start_date
    train_end_date = st.session_state.train_end_date
    env_params = st.session_state.env_params

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

                trial_model = UnifiedTradingAgent()

                # Train with current parameters
                metrics = trial_model.train(stock_names=stock_names,
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
                # Save best parameters
                st.session_state.ppo_params = study.best_params

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

        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            logger.exception("Hyperparameter optimization error")


def display_testing_interface(ppo_params, use_optuna_params=False):
    """
    Displays the testing interface and visualization options in a scrollable container
    """
    with st.container():
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
        st.session_state.test_start_date = test_start_date
        st.session_state.test_end_date = test_end_date

        st.markdown("""
            <style>
                .test-results {
                    max-height: 600px;
                    overflow-y: auto;
                    padding: 1rem;
                    border: 1px solid #e0e0e0;
                    border-radius: 4px;
                }
            </style>
        """,
                    unsafe_allow_html=True)

        if st.button("Test Model"):
            if not os.path.exists("trained_model.zip"):
                st.error("No trained model found. Please train a model first.")
            else:
                if use_optuna_params:
                    ppo_params = st.session_state.ppo_params

                test_results = st.session_state.model.test(
                    stock_names=st.session_state.stock_names,
                    start_date=st.session_state.test_start_date,
                    end_date=st.session_state.test_end_date,
                    env_params=st.session_state.env_params)

            # Display test metrics
            if test_results and 'metrics' in test_results:
                test_results_container = st.container()
                with test_results_container:
                    st.subheader("Test Results Analysis")

                    # Display test trade history
                    if 'info_history' in test_results:
                        TradingVisualizer.display_trade_history(
                            test_results['info_history'], "Test History",
                            "test_trade")

                    # Create tabs for different visualization aspects
                    metrics_tab, trades_tab, analysis_tab = st.tabs([
                        "Performance Metrics", "Trade Analysis",
                        "Technical Analysis"
                    ])

                    with metrics_tab:
                        st.subheader("Performance Metrics")

                    # Display parameters used for testing, automatically sorting into columns:
                    st.subheader("Parameters Used for Testing")
                    col1, col2, col3 = st.columns(3)
                    index_col = 0
                    all_cols = [col1, col2, col3]
                    for param, value in ppo_params.items():
                        with all_cols[index_col % 3]:
                            st.metric(param, value)
                            index_col += 1

                    # Now display the metrics:
                    metrics = test_results['metrics']

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sharpe Ratio",
                                  f"{metrics['sharpe_ratio']:.2f}")
                        st.metric("Max Drawdown",
                                  f"{metrics['max_drawdown']:.2%}")
                    with col2:
                        st.metric("Sortino Ratio",
                                  f"{metrics['sortino_ratio']:.2f}")
                        st.metric("Volatility", f"{metrics['volatility']:.2%}")
                    with col3:
                        if 'information_ratio' in metrics:
                            st.metric("Information Ratio",
                                      f"{metrics['information_ratio']:.2f}")
                        # Calculate and display total return
                        if 'portfolio_history' in test_results:
                            total_return = (
                                (test_results['portfolio_history'][-1] -
                                 test_results['portfolio_history'][0]) /
                                test_results['portfolio_history'][0])
                            st.metric("Total Return", f"{total_return:.2%}")

                    # Display performance charts
                    if 'combined_plot' in test_results:
                        st.plotly_chart(test_results['combined_plot'])
                    with trades_tab:
                        st.subheader("Trading Activity")

                        # Display discrete actions plot
                        if 'action_plot' in test_results:
                            st.plotly_chart(test_results['action_plot'],
                                            use_container_width=True)

                        # Display combined price and actions
                        if 'combined_plot' in test_results:
                            st.plotly_chart(test_results['combined_plot'],
                                            use_container_width=True)

                    with analysis_tab:
                        st.subheader("Technical Analysis")
                        if 'info_history' in test_results:
                            visualizer = TradingVisualizer()

                            # Show correlation analysis if multiple stocks
                            portfolio_data = st.session_state.model.data_handler.fetch_data(
                                st.session_state.stock_names,
                                st.session_state.test_start_date,
                                st.session_state.test_end_date)
                            if len(st.session_state.stock_names) > 1:
                                corr_fig = visualizer.plot_correlation_heatmap(
                                    portfolio_data)
                                st.plotly_chart(corr_fig,
                                                use_container_width=True)

                            # Show drawdown analysis
                            for symbol in st.session_state.stock_names:
                                drawdown_fig = visualizer.plot_performance_and_drawdown(
                                    portfolio_data, symbol)
                                st.plotly_chart(drawdown_fig,
                                                use_container_width=True)
