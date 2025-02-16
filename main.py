
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
    """Initialize Streamlit session state variables."""
    try:
        # Core components
        if 'log_messages' not in st.session_state:
            st.session_state.log_messages = []
        if 'ppo_params' not in st.session_state:
            st.session_state.ppo_params = None
        
        # Data handling
        if 'data_handler' not in st.session_state:
            data_handler = DataHandler()
            st.session_state.data_handler = data_handler
            
        # Ensure database connection
        if hasattr(st.session_state, 'data_handler'):
            if not st.session_state.data_handler.session or not st.session_state.data_handler.session.is_active:
                st.session_state.data_handler.get_session()
        
        # Model and stock data
        if 'model' not in st.session_state:
            st.session_state.model = UnifiedTradingAgent()
        if 'stock_list' not in st.session_state:
            st.session_state.stock_list = ['AAPL', 'MSFT']  # Fixed typo in AAPL
            
        # Training state
        if 'training_in_progress' not in st.session_state:
            st.session_state.training_in_progress = False
            
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        logging.error(f"Session state initialization failed: {str(e)}")


def main() -> None:
    init_session_state()
    
    if not check_system_health():
        st.stop()
        
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
def check_system_health():
    """Verify core system components are functioning."""
    try:
        # Check data handler
        if not st.session_state.data_handler or not st.session_state.data_handler.session:
            st.warning("⚠️ Database connection not initialized")
            return False
            
        # Check model
        if not st.session_state.model:
            st.warning("⚠️ Trading model not initialized")
            return False
            
        return True
        
    except Exception as e:
        st.error(f"System health check failed: {str(e)}")
        return False
