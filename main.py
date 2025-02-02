
import streamlit as st
from core.base_agent import UnifiedTradingAgent
from components.analysis_tab import display_tech_analysis_tab
from components.training_tab import display_training_tab
from components.database_tab import display_database_explorer
from data.data_handler import DataHandler

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def init_session_state() -> None:
    """
    Initialize Streamlit session state variables for persistent storage across reruns.

    Initializes:
        - log_messages: List[str] - Chronological log messages
        - ppo_params: Dict[str, Union[float, int, bool]] - PPO algorithm configuration
        - model: UnifiedTradingAgent - Trading agent model instance

    Implementation:
        The function checks for each required key in st.session_state and
        initializes it if missing. This ensures persistence across Streamlit reruns
        while avoiding reinitializing existing state.

    Example:
        ```python
        # Initialize state at app startup
        init_session_state()

        # Access state variables
        model = st.session_state.model
        logs = st.session_state.log_messages
        ```
    """
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    if 'ppo_params' not in st.session_state:
        st.session_state.ppo_params = None
    if 'model' not in st.session_state:
        st.session_state.model = UnifiedTradingAgent()
    if 'stock_list' not in st.session_state:
        st.session_state.stock_list = ['APPL', 'MSFT']
    if 'data_handler' not in st.session_state:
        st.session_state.data_handler = DataHandler()


def main() -> None:
    init_session_state()

    st.title("Trading Analysis and Agent Platform")

    # Create tabs for Technical Analysis, Model Training, and Database Explorer
    tab_training, tab_analysis, tab_database = st.tabs(
        ["Model Training", "Technical Analysis", "Database Explorer"])

    with tab_analysis:
        display_tech_analysis_tab()

    with tab_training:
        display_training_tab()

    with tab_database:
        display_database_explorer()


if __name__ == "__main__":
    main()
