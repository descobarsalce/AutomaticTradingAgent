"""
Core training functionality module
Encapsulates training-related functions from the training tab
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import streamlit as st
import pandas as pd
import optuna
import logging  # added import
from core.base_agent import UnifiedTradingAgent
from utils.callbacks import ProgressBarCallback
from utils.stock_utils import parse_stock_list
from core.visualization import TradingVisualizer

logger = logging.getLogger(__name__)  # added logger initialization

def initialize_training(stock_names: List[str], train_start_date: datetime,
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

    # Set logging level based on session state
    if st.session_state.get('enable_logging', False):
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL)


def execute_training(
        ppo_params: Dict[str, Any],
        progress_bar: Optional[st.progress] = None,
        status_placeholder: Optional[st.empty] = None) -> Dict[str, float]:
    """
    Execute model training with given parameters
    """
    progress_callback = None
    if progress_bar and status_placeholder:
        # Correct the timestep calculation: use end_date - start_date
        total_timesteps = (st.session_state.train_end_date - st.session_state.train_start_date).days
        logger.info(f"Training for {total_timesteps} timesteps")  # Debug training timesteps
        progress_callback = ProgressBarCallback(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
            status_placeholder=status_placeholder)

    # Get feature configuration from session state
    feature_config = st.session_state.get('feature_config', None)
    if feature_config:
        logger.info(f"Using feature config: {feature_config.get('use_feature_engineering', False)}")

    return st.session_state.model.train(
        stock_names=st.session_state.stock_names,
        start_date=st.session_state.train_start_date,
        end_date=st.session_state.train_end_date,
        env_params=st.session_state.env_params,
        ppo_params=ppo_params,
        callback=progress_callback,
        feature_config=feature_config)


def get_training_parameters(use_optuna_params: bool = False) -> Dict[str, Any]:
    """
    Get training parameters either from manual input or optuna optimization
    """
    if use_optuna_params:
        if st.session_state.ppo_params is not None:
            st.info("Using Optuna's optimized parameters")
            return st.session_state.ppo_params
        else:
            st.warning(
                "No Optuna parameters available. Please run hyperparameter tuning first."
            )
            return {}
    else:
        params = {}
        col3, col4 = st.columns(2)
        with col3:
            params['learning_rate'] = st.number_input("Learning Rate",
                                                      value=3e-4,
                                                      format="%.1e")
            params['n_steps'] = st.number_input("PPO Steps Per Update",
                                                value=512)
            params['batch_size'] = st.number_input("Batch Size", value=128)
            params['n_epochs'] = st.number_input("Number of Epochs", value=5)
        with col4:
            params['gamma'] = st.number_input("Gamma (Discount Factor)",
                                              value=0.99)
            params['clip_range'] = st.number_input("Clip Range", value=0.2)
            params['target_kl'] = st.number_input("Target KL Divergence",
                                                  value=0.05)
        return params


def display_training_metrics(metrics: Dict[str, float]) -> None:
    """Display training metrics from portfolio manager."""
    if not metrics:
        return

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

    with metrics_col1:
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        st.metric("Maximum Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
    with metrics_col2:
        st.metric("Return", f"{metrics.get('total_return', 0):.2%}")
        st.metric("Total Value", f"${metrics.get('total_value', 0):,.2f}")
    with metrics_col3:
        st.metric("Total Trades", metrics.get('total_trades', 0))
        st.metric("Cash Balance", f"${metrics.get('cash_balance', 0):,.2f}")


def run_training(ppo_params: Dict[str, Any]) -> None:
    """
    Executes the training process and displays results
    """
    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    metrics = execute_training(ppo_params, progress_bar, status_placeholder)

    if metrics:
        st.subheader("Parameters Used for Training")
        col1, col2, col3 = st.columns(3)
        index_col = 0
        all_cols = [col1, col2, col3]
        for param, value in ppo_params.items():
            with all_cols[index_col % 3]:
                st.metric(param, value)
                index_col += 1

        display_training_metrics(metrics)

    if hasattr(st.session_state.model.env, '_trade_history'):
        TradingVisualizer.display_trade_history(
            st.session_state.model.env._trade_history, "Training History",
            "training_trade")

    st.session_state.ppo_params = ppo_params
    st.success("Training completed and model saved!")
