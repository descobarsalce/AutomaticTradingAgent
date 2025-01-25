"""
Trading Agent Web Interface
A Streamlit-based dashboard for configuring and managing reinforcement learning trading agents.
"""

import streamlit as st
from datetime import datetime
from typing import List
from core.base_agent import UnifiedTradingAgent
from pages.technical_analysis import render_technical_analysis
from pages.model_training import render_model_training
from pages.database_explorer import render_database_explorer

def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if 'ppo_params' not in st.session_state:
        st.session_state.ppo_params = None
    if 'model' not in st.session_state:
        st.session_state.model = UnifiedTradingAgent()

def main() -> None:
    """Main application entry point."""
    init_session_state()

    st.title("Trading Analysis and Agent Platform")

    # Create tabs for Technical Analysis, Model Training, and Database Explorer
    tab_analysis, tab_training, tab_database = st.tabs([
        "Technical Analysis",
        "Model Training",
        "Database Explorer"
    ])

    with tab_analysis:
        render_technical_analysis(st.session_state.model)

    with tab_training:
        render_model_training()

    with tab_database:
        render_database_explorer()

if __name__ == "__main__":
    main()