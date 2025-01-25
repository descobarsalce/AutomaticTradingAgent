"""
Logging utilities for the trading platform
"""
import streamlit as st
import logging
from typing import List

class StreamlitLogHandler(logging.Handler):
    """
    Custom logging handler that redirects log messages to a container in the UI.
    """
    MAX_LOGS: int = 100

    def emit(self, record: logging.LogRecord) -> None:
        """
        Process a log record by formatting it and adding to the session state.

        Args:
            record: The log record to process
        """
        try:
            log_entry: str = self.format(record)
            if 'log_messages' in st.session_state:
                st.session_state.log_messages.append(log_entry)
                if len(st.session_state.log_messages) > self.MAX_LOGS:
                    st.session_state.log_messages = st.session_state.log_messages[-self.MAX_LOGS:]
            print(log_entry)  # Backup output
        except Exception as e:
            print(f"Logging error: {e}")

def init_session_state() -> None:
    """
    Initialize Streamlit session state variables for persistent storage across reruns.
    """
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    if 'ppo_params' not in st.session_state:
        st.session_state.ppo_params = None
    if 'model' not in st.session_state:
        from core.base_agent import UnifiedTradingAgent
        st.session_state.model = UnifiedTradingAgent()

def display_logs() -> None:
    """Display logs in a container within the UI."""
    if 'log_messages' in st.session_state and st.session_state.log_messages:
        with st.expander("System Logs", expanded=False):
            for log in st.session_state.log_messages:
                st.text(log)