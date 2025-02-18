
import streamlit as st
from core.base_agent import UnifiedTradingAgent
from components.analysis_tab import display_tech_analysis_tab
from components.training_tab import display_training_tab
from components.database_tab import display_database_explorer
from data.data_handler import DataHandler
from datetime import datetime, timedelta
import logging
from utils.logging_utils import StreamlitLogHandler

# Configure root logger with more concise format and INFO level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

def init_session_state() -> None:
    """Initialize Streamlit session state variables with timeout protection."""
    try:
        if 'initialized' in st.session_state:
            return

        # Set initialization timeout
        start_time = datetime.now()
        timeout = timedelta(seconds=30)
        
        logger.info("Setting session state initialized flag")
        st.session_state.initialized = True
        
        if (datetime.now() - start_time) > timeout:
            raise TimeoutError("Session state initialization timed out")
            
    except Exception as e:
        logger.error(f"Session state initialization error: {str(e)}")
        # raise
        # st.session_state.initialized = True
        
        logger.info("Initializing log messages array")
        if 'log_messages' not in st.session_state:
            st.session_state.log_messages = []
            
        logger.info("Checking PPO parameters")
        if 'ppo_params' not in st.session_state:
            logger.info("Initializing PPO parameters to None")
            st.session_state.ppo_params = None
        
        logger.info("Initializing DataHandler")
        if 'data_handler' not in st.session_state:
            try:
                logger.info("Creating new DataHandler instance")
                data_handler = DataHandler()
                st.session_state.data_handler = data_handler
                logger.info("DataHandler initialized successfully")
            except Exception as e:
                logger.error(f"DataHandler initialization failed: {str(e)}")
                raise
            
        logger.info("Checking database connection")
        if hasattr(st.session_state, 'data_handler'):
            if not st.session_state.data_handler.session or not st.session_state.data_handler.session.is_active:
                logger.warning("Database session inactive, attempting reconnection")
                try:
                    st.session_state.data_handler.get_session()
                    logger.info("Database connection re-established")
                except Exception as e:
                    logger.error(f"Database reconnection failed: {str(e)}")
                    raise
        
        logger.info("Initializing trading model")
        if 'model' not in st.session_state:
            logger.info("Creating new UnifiedTradingAgent")
            st.session_state.model = UnifiedTradingAgent()
            
        logger.info("Setting default stock list")
        if 'stock_list' not in st.session_state:
            st.session_state.stock_list = ['AAPL', 'MSFT']
            
        logger.info("Initializing training state")
        if 'training_in_progress' not in st.session_state:
            st.session_state.training_in_progress = False
            
        logger.info("✅ Session state initialization completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Session state initialization failed: {str(e)}")
        raise

def check_system_health() -> bool:
    """Verify core system components are functioning."""
    logger.info("🔍 Starting system health check")
    
    try:
        if 'data_handler' not in st.session_state:
            logger.info("Initializing missing DataHandler")
            st.session_state.data_handler = DataHandler()
            
        logger.info("Verifying database connection")
        if not st.session_state.data_handler.session or not st.session_state.data_handler.session.is_active:
            logger.warning("Database session inactive, attempting reconnection")
            st.session_state.data_handler.get_session()
            
        if not st.session_state.data_handler.session.is_active:
            logger.error("Database session is not active after reconnection attempt")
            return False
            
        logger.info("Checking trading agent initialization")
        if not getattr(st.session_state, 'model', None):
            logger.info("Initializing missing trading agent")
            st.session_state.model = UnifiedTradingAgent()
            
        logger.info("✅ System health check completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ System health check failed: {str(e)}")
        return False

def main() -> None:
    """Main application entry point with timeout protection."""
    start_time = datetime.now()
    logger.info("Starting main application")
    
    try:
        # Add timeout protection
        if (datetime.now() - start_time) > timedelta(seconds=60):
            st.error("Application startup timed out. Please refresh the page.")
            return
            
        init_session_state()
        
        logger.info("Performing system health check")
        if not check_system_health():
            logger.error("System health check failed")
            st.error("System health check failed. Please check the logs.")
            st.stop()
            return
        
        logger.info("Setting up main UI")
        st.title("Trading Analysis and Agent Platform")

        logger.info("Creating application tabs")
        tab_training, tab_analysis, tab_database = st.tabs(
            ["Model Training", "Technical Analysis", "Database Explorer"])

        logger.info("Initializing Technical Analysis tab")
        with tab_analysis:
            try:
                display_tech_analysis_tab()
                logger.info("Technical Analysis tab loaded successfully")
            except Exception as e:
                logger.error(f"Error in Technical Analysis tab: {str(e)}")
                st.error(f"Error loading Technical Analysis tab: {str(e)}")

        logger.info("Initializing Model Training tab")
        with tab_training:
            try:
                display_training_tab()
                logger.info("Model Training tab loaded successfully")
            except Exception as e:
                logger.error(f"Error in Model Training tab: {str(e)}")
                st.error(f"Error loading Model Training tab: {str(e)}")

        logger.info("Initializing Database Explorer tab")
        with tab_database:
            try:
                display_database_explorer()
                logger.info("Database Explorer tab loaded successfully")
            except Exception as e:
                logger.error(f"Error in Database Explorer tab: {str(e)}")
                st.error(f"Error loading Database Explorer tab: {str(e)}")

        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✨ Application initialization completed in {execution_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"❌ Critical application error: {str(e)}")
        st.error(f"A critical error occurred: {str(e)}")

if __name__ == "__main__":
    main()
