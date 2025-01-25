"""
Model Training Module
Handles the model training functionality for the trading platform.
"""

import streamlit as st
from datetime import datetime, timedelta
import optuna
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, Optional

def hyperparameter_tuning(stock_name: str, train_start_date: datetime,
                         train_end_date: datetime,
                         env_params: Dict[str, Any]) -> None:
    """Implementation of hyperparameter tuning interface and functionality."""
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
            help="Metric to optimize during hyperparameter search"
        )
    
    if st.button("Start Hyperparameter Tuning"):
        run_hyperparameter_optimization(
            stock_name, train_start_date, train_end_date,
            env_params, trials_number, pruning_enabled,
            optimization_metric, lr_min, lr_max, steps_min, steps_max,
            batch_min, batch_max, epochs_min, epochs_max,
            gamma_min, gamma_max, gae_min, gae_max
        )

def run_hyperparameter_optimization(stock_name: str, train_start_date: datetime,
                                  train_end_date: datetime, env_params: Dict[str, Any],
                                  trials_number: int, pruning_enabled: bool,
                                  optimization_metric: str, *params) -> None:
    """Run the hyperparameter optimization process."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create study with pruning
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner() if pruning_enabled else None
    )
    
    def objective(trial: optuna.Trial) -> float:
        try:
            ppo_params = {
                'learning_rate':
                trial.suggest_loguniform('learning_rate', params[0], params[1]),
                'n_steps':
                trial.suggest_int('n_steps', params[2], params[3]),
                'batch_size':
                trial.suggest_int('batch_size', params[4], params[5]),
                'n_epochs':
                trial.suggest_int('n_epochs', params[6], params[7]),
                'gamma':
                trial.suggest_uniform('gamma', params[8], params[9]),
                'gae_lambda':
                trial.suggest_uniform('gae_lambda', params[10], params[11]),
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
                ppo_params=ppo_params
            )
            
            trial_value = metrics.get(optimization_metric, float('-inf'))
            progress = (trial.number + 1) / trials_number
            progress_bar.progress(progress)
            
            return trial_value
            
        except Exception as e:
            st.error(f"Error in trial {trial.number}: {str(e)}")
            return float('-inf')
    
    try:
        study.optimize(objective, n_trials=trials_number)
        display_optimization_results(study, optimization_metric)
        
    except Exception as e:
        st.error(f"Optimization failed: {str(e)}")

def display_optimization_results(study: optuna.study.Study, 
                               optimization_metric: str) -> None:
    """Display the results of hyperparameter optimization."""
    st.success("Hyperparameter tuning completed!")
    
    # Create detailed results dataframe
    trials_df = pd.DataFrame([{
        'Trial': t.number,
        'Value': t.value,
        **t.params
    } for t in study.trials if t.value is not None])
    
    # Display results in tabs
    tab1, tab2, tab3 = st.tabs([
        "Best Parameters",
        "Optimization History",
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
            go.Scatter(
                x=trials_df.index,
                y=trials_df['Value'],
                mode='lines+markers',
                name='Trial Value'
            )
        )
        history_fig.update_layout(
            title='Optimization History',
            xaxis_title='Trial Number',
            yaxis_title=optimization_metric.replace('_', ' ').title()
        )
        st.plotly_chart(history_fig)
    
    with tab3:
        st.subheader("Parameter Importance")
        importance_dict = optuna.importance.get_param_importances(study)
        importance_df = pd.DataFrame({
            'Parameter': list(importance_dict.keys()),
            'Importance': list(importance_dict.values())
        }).sort_values('Importance', ascending=True)
        
        importance_fig = go.Figure()
        importance_fig.add_trace(
            go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Parameter'],
                orientation='h'
            )
        )
        importance_fig.update_layout(
            title='Parameter Importance Analysis',
            xaxis_title='Relative Importance',
            yaxis_title='Parameter',
            height=400
        )
        st.plotly_chart(importance_fig)
    
    # Save and provide download for results
    trials_df.to_csv('hyperparameter_tuning_results.csv', index=False)
    st.download_button(
        "Download Complete Results CSV",
        trials_df.to_csv(index=False),
        "hyperparameter_tuning_results.csv",
        "text/csv"
    )
    
    # Save best parameters to session state
    st.session_state.ppo_params = study.best_params

def render_model_training() -> None:
    """Render the model training interface."""
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

    # Date selection for training
    st.subheader("Training Period")
    train_col1, train_col2 = st.columns(2)
    with train_col1:
        train_start_date = datetime.combine(
            st.date_input(
                "Training Start Date",
                value=datetime.now() - timedelta(days=365 * 5)
            ),
            datetime.min.time()
        )
    with train_col2:
        train_end_date = datetime.combine(
            st.date_input(
                "Training End Date",
                value=datetime.now() - timedelta(days=365 + 1)
            ),
            datetime.min.time()
        )

    tab1, tab2 = st.tabs(["Manual Parameters", "Hyperparameter Tuning"])

    with tab1:
        # Manual parameter selection
        st.header("Agent Parameters")

        use_optuna_params = st.checkbox(
            "Use Optuna Optimized Parameters",
            value=False
        )

        if use_optuna_params and hasattr(st.session_state, 'ppo_params') and st.session_state.ppo_params is not None:
            st.info("Using Optuna's optimized parameters")
            # Display Optuna parameters as read-only
            col3, col4 = st.columns(2)
            with col3:
                st.text(
                    f"Learning Rate: {st.session_state.ppo_params['learning_rate']:.2e}"
                )
                st.text(
                    f"PPO Steps: {st.session_state.ppo_params['n_steps']}"
                )
                st.text(
                    f"Batch Size: {st.session_state.ppo_params['batch_size']}"
                )
                st.text(
                    f"Number of Epochs: {st.session_state.ppo_params['n_epochs']}"
                )
            with col4:
                st.text(
                    f"Gamma: {st.session_state.ppo_params['gamma']:.4f}"
                )
                st.text(
                    f"GAE Lambda: {st.session_state.ppo_params['gae_lambda']:.4f}"
                )
        else:
            # Manual parameter input
            col3, col4 = st.columns(2)
            with col3:
                learning_rate = st.number_input(
                    "Learning Rate",
                    value=3e-4,
                    format="%.0e",
                    key="manual_lr"
                )
                n_steps = st.number_input(
                    "PPO Steps",
                    value=2048,
                    step=64,
                    key="manual_steps"
                )
                batch_size = st.number_input(
                    "Batch Size",
                    value=64,
                    step=32,
                    key="manual_batch"
                )
            with col4:
                n_epochs = st.number_input(
                    "Training Epochs",
                    value=10,
                    step=1,
                    key="manual_epochs"
                )
                gamma = st.number_input(
                    "Gamma",
                    value=0.99,
                    step=0.01,
                    format="%.3f",
                    key="manual_gamma"
                )
                gae_lambda = st.number_input(
                    "GAE Lambda",
                    value=0.95,
                    step=0.01,
                    format="%.3f",
                    key="manual_gae"
                )

            if st.button("Start Training"):
                manual_params = {
                    'learning_rate': learning_rate,
                    'n_steps': n_steps,
                    'batch_size': batch_size,
                    'n_epochs': n_epochs,
                    'gamma': gamma,
                    'gae_lambda': gae_lambda
                }

                try:
                    metrics = st.session_state.model.train(
                        stock_name=stock_name,
                        start_date=train_start_date,
                        end_date=train_end_date,
                        env_params=env_params,
                        ppo_params=manual_params
                    )
                    st.success("Training completed successfully!")
                    st.write("Training Metrics:", metrics)
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")

    with tab2:
        hyperparameter_tuning(
            stock_name, train_start_date, train_end_date, env_params
        )