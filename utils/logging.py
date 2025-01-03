import logging
import streamlit as st
from datetime import datetime
from typing import Optional

class StreamlitLogHandler(logging.Handler):
    """A robust logging handler that writes logs to Streamlit's UI."""

    def __init__(self):
        super().__init__()
        # Initialize logs list in session state if not present
        if not hasattr(st.session_state, 'logs'):
            st.session_state.logs = []

    def emit(self, record):
        try:
            # Format the log entry with timestamp
            timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
            msg = record.getMessage()
            log_entry = f"{timestamp} - {record.levelname} - {msg}"

            # Store the log entry with its level and formatted message
            st.session_state.logs.append({
                'level': record.levelname,
                'message': log_entry
            })

            # Trim logs if they exceed 1000 entries
            if len(st.session_state.logs) > 1000:
                st.session_state.logs = st.session_state.logs[-1000:]

        except Exception as e:
            print(f"Error in StreamlitLogHandler.emit: {str(e)}")

def setup_logging(level: str = 'INFO') -> None:
    """Configure logging with both console and Streamlit handlers."""
    try:
        # Create handlers
        streamlit_handler = StreamlitLogHandler()
        console_handler = logging.StreamHandler()

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        streamlit_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.handlers = []  # Remove existing handlers
        root_logger.addHandler(streamlit_handler)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(level.upper())

    except Exception as e:
        print(f"Error in setup_logging: {str(e)}")
        raise

def display_logs(container: Optional[st.container] = None) -> None:
    """Display logs in the Streamlit UI with error handling."""
    try:
        # Use provided container or st directly
        target = container if container else st

        # Ensure logs exist in session state
        if hasattr(st.session_state, 'logs'):
            # Display logs with appropriate styling
            for log in reversed(st.session_state.logs):
                if log['level'] in ('ERROR', 'CRITICAL'):
                    target.error(log['message'])
                elif log['level'] == 'WARNING':
                    target.warning(log['message'])
                else:
                    target.info(log['message'])

    except Exception as e:
        print(f"Error in display_logs: {str(e)}")
        if container:
            container.error(f"Error displaying logs: {str(e)}")