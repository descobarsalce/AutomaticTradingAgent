"""
Testing Interface Tab Component
Dedicated tab for testing trained models with optimized parameters
"""
import streamlit as st
import logging
from datetime import datetime, timedelta

from core.testing_functions import display_testing_interface
from core.hyperparameter_search import load_best_params

logger = logging.getLogger(__name__)


def display_testing_tab():
    """
    Renders the dedicated testing interface tab
    This tab uses the best parameters from hyperparameter optimization
    """
    st.header("Model Testing Interface")

    # Check if we have the required session state variables
    if 'model' not in st.session_state:
        st.error("⚠️ No model initialized. Please initialize the system from the 'Model Training' tab first.")
        return

    if 'stock_names' not in st.session_state:
        st.error("⚠️ No stock names configured. Please configure stocks in the 'Model Training' tab first.")
        return

    if 'env_params' not in st.session_state:
        st.error("⚠️ No environment parameters configured. Please configure environment in the 'Model Training' tab first.")
        return

    # Load best parameters from hyperparameter optimization
    best_params_data = load_best_params()

    if best_params_data is None:
        st.warning(
            "⚠️ No optimized parameters found.\n\n"
            "Please run hyperparameter tuning in the 'Model Training' → 'Hyperparameter Tuning' tab first."
        )
        return

    # Display parameter information
    st.subheader("Current Best Parameters")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        phase_info = best_params_data.get('phase')
        if phase_info:
            phase_name = "Exploration" if phase_info == 1 else "Exploitation"
            st.metric("Optimization Phase", f"Phase {phase_info} ({phase_name})")
        else:
            st.metric("Optimization Phase", "Single Phase")

    with col2:
        value = best_params_data.get('value', 'N/A')
        if isinstance(value, (int, float)):
            st.metric("Best Value", f"{value:.4f}")
        else:
            st.metric("Best Value", value)

    with col3:
        timestamp = best_params_data.get('timestamp')
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                st.caption(f"Optimized on:\n{dt.strftime('%Y-%m-%d %H:%M')}")
            except:
                st.caption("Timestamp unavailable")

    # Display parameters in an expandable section
    with st.expander("View Best Parameters", expanded=False):
        params = best_params_data.get('params', {})
        if params:
            col1, col2 = st.columns(2)

            param_list = list(params.items())
            mid_point = len(param_list) // 2

            with col1:
                for param, value in param_list[:mid_point]:
                    if isinstance(value, float) and param == 'learning_rate':
                        st.metric(param, f"{value:.2e}")
                    elif isinstance(value, float):
                        st.metric(param, f"{value:.4f}")
                    else:
                        st.metric(param, str(value))

            with col2:
                for param, value in param_list[mid_point:]:
                    if isinstance(value, float) and param == 'learning_rate':
                        st.metric(param, f"{value:.2e}")
                    elif isinstance(value, float):
                        st.metric(param, f"{value:.4f}")
                    else:
                        st.metric(param, str(value))
        else:
            st.warning("No parameters found")

    # Refresh button
    if st.button("🔄 Refresh Parameters from Latest Optimization"):
        best_params_data = load_best_params()
        if best_params_data is not None:
            st.session_state.ppo_params = best_params_data['params']
            st.success("✅ Parameters refreshed from latest hyperparameter optimization!")
            st.rerun()
        else:
            st.error("❌ Failed to load parameters")

    st.markdown("---")

    # Store parameters in session state
    st.session_state.ppo_params = best_params_data.get('params', {})

    # Display the testing interface
    display_testing_interface(
        st.session_state.model,
        st.session_state.stock_names,
        st.session_state.env_params,
        st.session_state.ppo_params,
        use_optuna_params=True
    )
