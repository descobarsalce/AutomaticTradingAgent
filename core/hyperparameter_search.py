
"""
Hyperparameter Search Module
Handles Optuna-based hyperparameter optimization
"""
import streamlit as st
import optuna
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import logging
from typing import Dict, Any, Optional
from core.base_agent import UnifiedTradingAgent

logger = logging.getLogger(__name__)

def create_parameter_ranges() -> Dict[str, Any]:
    """Get parameter ranges from user input"""
    col1, col2 = st.columns(2)
    with col1:
        lr_min = st.number_input("Learning Rate Min", value=1e-5, format="%.1e")
        lr_max = st.number_input("Learning Rate Max", value=5e-4, format="%.1e")
        steps_min = st.number_input("Steps Min", value=512, step=64)
        steps_max = st.number_input("Steps Max", value=2048, step=64)
        batch_min = st.number_input("Batch Size Min", value=64, step=32)
        batch_max = st.number_input("Batch Size Max", value=512, step=32)

    with col2:
        epochs_min = st.number_input("Training Epochs Min", value=3, step=1)
        epochs_max = st.number_input("Training Epochs Max", value=10, step=1)
        gamma_min = st.number_input("Gamma Min", value=0.90, step=0.01, format="%.3f")
        gamma_max = st.number_input("Gamma Max", value=0.999, step=0.001, format="%.3f")
        gae_min = st.number_input("GAE Lambda Min", value=0.90, step=0.01, format="%.2f")
        gae_max = st.number_input("GAE Lambda Max", value=0.99, step=0.01, format="%.2f")

    return {
        'lr': (lr_min, lr_max),
        'steps': (steps_min, steps_max),
        'batch': (batch_min, batch_max),
        'epochs': (epochs_min, epochs_max),
        'gamma': (gamma_min, gamma_max),
        'gae': (gae_min, gae_max)
    }

def run_hyperparameter_optimization(stock_names: list,
                                  train_start_date: datetime,
                                  train_end_date: datetime,
                                  env_params: Dict[str, Any],
                                  param_ranges: Dict[str, Any],
                                  trials_number: int,
                                  optimization_metric: str,
                                  progress_bar,
                                  status_text,
                                  pruning_enabled: bool = True) -> optuna.Study:
    """Run hyperparameter optimization using Optuna"""
    
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner() if pruning_enabled else None)

    def objective(trial: optuna.Trial) -> float:
        try:
            ppo_params = {
                'learning_rate': trial.suggest_loguniform('learning_rate', 
                                                        param_ranges['lr'][0], 
                                                        param_ranges['lr'][1]),
                'n_steps': trial.suggest_int('n_steps', 
                                           param_ranges['steps'][0], 
                                           param_ranges['steps'][1]),
                'batch_size': trial.suggest_int('batch_size', 
                                              param_ranges['batch'][0], 
                                              param_ranges['batch'][1]),
                'n_epochs': trial.suggest_int('n_epochs', 
                                            param_ranges['epochs'][0], 
                                            param_ranges['epochs'][1]),
                'gamma': trial.suggest_uniform('gamma', 
                                             param_ranges['gamma'][0], 
                                             param_ranges['gamma'][1]),
                'gae_lambda': trial.suggest_uniform('gae_lambda', 
                                                  param_ranges['gae'][0], 
                                                  param_ranges['gae'][1]),
            }

            status_text.text(f"Trial {trial.number + 1}/{trials_number}: Testing parameters {ppo_params}")
            trial_model = UnifiedTradingAgent()
            metrics = trial_model.train(stock_names=stock_names,
                                      start_date=train_start_date,
                                      end_date=train_end_date,
                                      env_params=env_params,
                                      ppo_params=ppo_params)

            trial_value = metrics.get(optimization_metric, float('-inf'))
            progress = (trial.number + 1) / trials_number
            progress_bar.progress(progress)

            return trial_value

        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            return float('-inf')

    study.optimize(objective, n_trials=trials_number)
    return study

def save_best_params(params: Dict[str, Any], value: float) -> None:
    """Save best parameters to a file"""
    import json
    best_params = {'params': params, 'value': value}
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f)

def load_best_params() -> Optional[Dict[str, Any]]:
    """Load best parameters from file"""
    import json
    try:
        with open('best_hyperparameters.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def display_optimization_results(study: optuna.Study) -> None:
    """Display optimization results using Streamlit"""
    # Save best parameters
    save_best_params(study.best_params, study.best_value)
    
    trials_df = pd.DataFrame([{
        'Trial': t.number,
        'Value': t.value,
        **t.params
    } for t in study.trials if t.value is not None])

    tab1, tab2, tab3 = st.tabs(["Best Parameters", "Optimization History", "Parameter Importance"])

    with tab1:
        st.subheader("Best Configuration Found")
        for param, value in study.best_params.items():
            if param == 'learning_rate':
                st.metric(f"Best {param}", f"{value:.2e}")
            elif param in ['gamma', 'gae_lambda']:
                st.metric(f"Best {param}", f"{value:.4f}")
            else:
                st.metric(f"Best {param}", f"{int(value)}")
        st.metric("Best Value", f"{study.best_value:.6f}")
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
            yaxis_title='Metric Value')
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
            go.Bar(x=importance_df['Importance'],
                  y=importance_df['Parameter'],
                  orientation='h'))
        importance_fig.update_layout(
            title='Parameter Importance Analysis',
            xaxis_title='Relative Importance',
            yaxis_title='Parameter',
            height=400)
        st.plotly_chart(importance_fig)

    trials_df.to_csv('hyperparameter_tuning_results.csv', index=False)
    st.download_button(
        "Download Complete Results CSV",
        trials_df.to_csv(index=False),
        "hyperparameter_tuning_results.csv",
        "text/csv")
