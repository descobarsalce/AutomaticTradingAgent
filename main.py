"""
Trading Agent Web Interface
A Streamlit-based dashboard for configuring and managing reinforcement learning trading agents.

System Architecture:
- Web Interface Layer (Streamlit): Provides UI for configuration and visualization
- Agent Layer (UnifiedTradingAgent): Implements trading strategy and training logic
- Environment Layer (SimpleTradingEnv): Simulates market interactions
- Data Layer: Handles market data acquisition and preprocessing
"""

import streamlit as st
import logging
from datetime import datetime
from utils.logging_utils import StreamlitLogHandler, init_session_state
from components.analysis_tab import display_analysis_tab
from components.training_tab import display_training_tab

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main() -> None:
    """
    Main application entry point
    Sets up the application layout and manages tab navigation
    """
    init_session_state()

    # Add StreamlitLogHandler if not already added
    if not any(isinstance(handler, StreamlitLogHandler) for handler in logger.handlers):
        handler = StreamlitLogHandler()
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    st.title("Trading Analysis and Agent Platform")

    # Main navigation tabs
    tab_analysis, tab_training = st.tabs(["Technical Analysis", "Agent Training"])

    # Display content based on selected tab
    with tab_analysis:
        display_analysis_tab(st.session_state.model)

    with tab_training:
        display_training_tab()

if __name__ == "__main__":
    main()