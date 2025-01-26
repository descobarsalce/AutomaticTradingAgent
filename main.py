
import streamlit as st
from core.base_agent import UnifiedTradingAgent
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from utils.logging_utils import StreamlitLogHandler, init_session_state


def main() -> None:
    init_session_state()

    st.title("Trading Analysis and Agent Platform")

    # Create tabs for Technical Analysis, Model Training, and Database Explorer
    tab_training, tab_analysis, tab_database = st.tabs(
        ["Model Training", "Technical Analysis", "Database Explorer"])

    with tab_analysis:
        from components.analysis_tab import display_analysis_tab
        display_analysis_tab(st.session_state.model)

    with tab_training:
        from components.training_tab import display_training_tab
        display_training_tab()

    with tab_database:
        from components.database_tab import display_database_explorer
        display_database_explorer()


if __name__ == "__main__":
    main()
