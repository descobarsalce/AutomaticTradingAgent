"""
Logging utilities for the trading platform
"""
import streamlit as st
import logging
from typing import List, Dict, Union, Any
from core.base_agent import UnifiedTradingAgent


class StreamlitLogHandler(logging.Handler):
    """
    Custom logging handler that redirects log messages to Streamlit's interface.

    Maintains a fixed-size buffer of recent log messages in the session state.
    Implements the observer pattern to capture logs from all components.

    Attributes:
        MAX_LOGS: int - Maximum number of logs to maintain in memory (100)
        format: Callable - Log message formatter function

    Implementation Details:
        - Uses Streamlit session state for persistence
        - Implements circular buffer behavior for log storage
        - Thread-safe for concurrent logging

    Example:
        ```python
        # Setup logging
        handler = StreamlitLogHandler()
        logger.addHandler(handler)

        # Log messages will appear in Streamlit sidebar
        logger.info("Training started...")
        ```
    """

    MAX_LOGS: int = 100

    def emit(self, record: logging.LogRecord) -> None:
        """
        Process a log record by formatting it and adding to the session state.

        Args:
            record (logging.LogRecord): The log record containing:
                - msg: str - The log message
                - levelno: int - Logging level number
                - created: float - Time when the log was created
                - args: tuple - Message format arguments

        Implementation:
            1. Formats the log record using the handler's formatter
            2. Appends to session state log buffer
            3. Maintains maximum log count by removing oldest entries
            4. Handles formatting exceptions gracefully

        Raises:
            Exception: If logging fails, error is printed but not propagated
        """
        try:
            log_entry: str = self.format(record)
            if 'log_messages' in st.session_state:
                st.session_state.log_messages.append(log_entry)
                if len(st.session_state.log_messages) > self.MAX_LOGS:
                    st.session_state.log_messages = st.session_state.log_messages[
                        -self.MAX_LOGS:]
            print(log_entry)  # Backup output
        except Exception as e:
            print(f"Logging error: {e}")

    @staticmethod
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
