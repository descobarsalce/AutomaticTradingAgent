
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
    """Initialize Streamlit session state variables with detailed logging."""
    if 'initialized' in st.session_state:
        logger.info("🔄 Session already initialized, skipping initialization")
        return

    start_time = datetime.now()
    logger.info("🚀 Starting session state initialization...")
    
    # Configure logging with more detail
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    try:
        st.session_state.initialized = True
        logger.info("📊 Setting up logging configuration...")
        logging.basicConfig(
            level=logging.INFO,  # Reduced logging level
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        
        logger.info("🔄 Initializing core components...")
        if 'log_messages' not in st.session_state:
            st.session_state.log_messages = []
            
        logger.info(f"Session initialization time: {(datetime.now() - start_time).total_seconds():.2f}s")
        if 'ppo_params' not in st.session_state:
            st.session_state.ppo_params = None
        
        # Data handling with enhanced logging
        if 'data_handler' not in st.session_state:
            logger.info("📊 Initializing DataHandler...")
            try:
                data_handler = DataHandler()
                st.session_state.data_handler = data_handler
                logger.info("✅ DataHandler initialized successfully")
            except Exception as e:
                logger.error(f"❌ DataHandler initialization failed: {str(e)}")
                raise
            
        # Ensure database connection with detailed logging
        if hasattr(st.session_state, 'data_handler'):
            logger.info("🔍 Checking database connection...")
            if not st.session_state.data_handler.session or not st.session_state.data_handler.session.is_active:
                logger.warning("⚠️ Database session inactive or missing, attempting to reconnect...")
                try:
                    st.session_state.data_handler.get_session()
                    logger.info("✅ Database connection re-established")
                except Exception as e:
                    logger.error(f"❌ Database reconnection failed: {str(e)}")
                    raise
        
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
    logger.info("🎯 Starting main application")
    try:
        init_session_state()
        logger.info("✅ Session state initialized")
        
        if not check_system_health():
            logger.error("❌ System health check failed")
            st.stop()
            return
        
        logger.info("🎨 Initializing user interface")
        st.title("Trading Analysis and Agent Platform")

        logger.info("📑 Creating application tabs")
    # Create tabs for Technical Analysis, Model Training, and Database Explorer
    tab_training, tab_analysis, tab_database = st.tabs(
        ["Model Training", "Technical Analysis", "Database Explorer"])

    logger.info("📊 Initializing Technical Analysis tab")
    with tab_analysis:
        try:
            display_tech_analysis_tab()
            logger.info("✅ Technical Analysis tab loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error in Technical Analysis tab: {str(e)}")
            raise

    logger.info("🤖 Initializing Model Training tab")
    with tab_training:
        try:
            display_training_tab()
            logger.info("✅ Model Training tab loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error in Model Training tab: {str(e)}")
            raise

    logger.info("🗄️ Initializing Database Explorer tab")
    with tab_database:
        try:
            display_database_explorer()
            logger.info("✅ Database Explorer tab loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error in Database Explorer tab: {str(e)}")
            raise
            
    logger.info("✨ Application initialization completed successfully")


if __name__ == "__main__":
    main()
def check_system_health():
    """Verify core system components are functioning."""
    try:
        logger.info("🔍 Starting system health check...")
        
        # Initialize components if missing
        if 'data_handler' not in st.session_state:
            logger.info("📊 Initializing DataHandler...")
            st.session_state.data_handler = DataHandler()
            logger.info("✅ DataHandler initialized")
            
        # Ensure active database connection
        if not st.session_state.data_handler.session or not st.session_state.data_handler.session.is_active:
            logger.info("🔌 Establishing database session...")
            st.session_state.data_handler.get_session()
            
        # Verify database connection
        if not st.session_state.data_handler.session.is_active:
            logger.error("❌ Database session is not active")
            st.error("⚠️ Unable to establish database connection")
            return False
            
        # Check model initialization
        if not getattr(st.session_state, 'model', None):
            logger.info("🤖 Initializing trading agent...")
            st.session_state.model = UnifiedTradingAgent()
            logger.info("✅ Trading agent initialized")
            
        logger.info("✨ System health check completed successfully")
        return True
        
    except Exception as e:
        st.error(f"System initialization failed: {str(e)}")
        logger.exception("Critical system initialization error")
        return False
