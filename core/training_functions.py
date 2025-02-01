
"""
Core training functionality module
Encapsulates training-related functions from the training tab
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import streamlit as st
import pandas as pd
import optuna
from core.base_agent import UnifiedTradingAgent
from utils.callbacks import ProgressBarCallback
from utils.stock_utils import parse_stock_list

def initialize_training(stock_names: List[str], 
                       train_start_date: datetime,
                       train_end_date: datetime,
                       env_params: Dict[str, Any]) -> None:
    """
    Initialize training environment and model
    """
    if 'model' not in st.session_state:
        st.session_state.model = UnifiedTradingAgent()
        
    st.session_state.stock_names = stock_names
    st.session_state.train_start_date = train_start_date
    st.session_state.train_end_date = train_end_date
    st.session_state.env_params = env_params

def execute_training(ppo_params: Dict[str, Any],
                    progress_bar: Optional[st.progress] = None,
                    status_placeholder: Optional[st.empty] = None) -> Dict[str, float]:
    """
    Execute model training with given parameters
    """
    progress_callback = None
    if progress_bar and status_placeholder:
        progress_callback = ProgressBarCallback(
            total_timesteps=(st.session_state.train_start_date -
                           st.session_state.train_end_date).days,
            progress_bar=progress_bar,
            status_placeholder=status_placeholder)

    return st.session_state.model.train(
        stock_names=st.session_state.stock_names,
        start_date=st.session_state.train_start_date,
        end_date=st.session_state.train_end_date,
        env_params=st.session_state.env_params,
        ppo_params=ppo_params,
        callback=progress_callback)

def get_training_parameters(use_optuna_params: bool = False) -> Dict[str, Any]:
    """
    Get training parameters either from manual input or optuna optimization
    """
    if use_optuna_params:
        if st.session_state.ppo_params is not None:
            st.info("Using Optuna's optimized parameters")
            return st.session_state.ppo_params
        else:
            st.warning("No Optuna parameters available. Please run hyperparameter tuning first.")
            return {}
    else:
        params = {}
        col3, col4 = st.columns(2)
        with col3:
            params['learning_rate'] = st.number_input("Learning Rate", value=3e-4, format="%.1e")
            params['n_steps'] = st.number_input("PPO Steps Per Update", value=512)
            params['batch_size'] = st.number_input("Batch Size", value=128)
            params['n_epochs'] = st.number_input("Number of Epochs", value=5)
        with col4:
            params['gamma'] = st.number_input("Gamma (Discount Factor)", value=0.99)
            params['clip_range'] = st.number_input("Clip Range", value=0.2)
            params['target_kl'] = st.number_input("Target KL Divergence", value=0.05)
        return params

def display_training_metrics(metrics: Dict[str, float]) -> None:
    """
    Display training metrics in a formatted layout
    """
    if not metrics:
        return
        
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
