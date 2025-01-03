import logging
import streamlit as st
from datetime import datetime
from typing import Optional, Dict, Any

class StreamlitLogHandler(logging.Handler):
    """A robust logging handler that writes logs to Streamlit's UI."""

    def __init__(self):
        super().__init__()
        if 'logs' not in st.session_state:
            st.session_state.logs = []

    def emit(self, record):
        try:
            # Format the log entry
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
                'level': record.levelname,
                'message': record.getMessage(),
                'logger_name': record.name
            }

            # Store the log entry
            st.session_state.logs.append(log_entry)

            # Keep only the last 1000 logs
            if len(st.session_state.logs) > 1000:
                st.session_state.logs = st.session_state.logs[-1000:]

        except Exception as e:
            print(f"Error in StreamlitLogHandler.emit: {str(e)}")

def setup_logging(level: str = 'INFO') -> None:
    """Configure logging with both console and Streamlit handlers."""
    try:
        # Remove all existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers = []

        # Create handlers
        streamlit_handler = StreamlitLogHandler()
        console_handler = logging.StreamHandler()

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        streamlit_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Set log level
        log_level = getattr(logging, level.upper(), logging.INFO)
        root_logger.setLevel(log_level)

        # Add handlers
        root_logger.addHandler(streamlit_handler)
        root_logger.addHandler(console_handler)

        # Log setup completion
        logging.info(f"Logging system initialized with level: {level}")

    except Exception as e:
        print(f"Error in setup_logging: {str(e)}")
        raise

def display_logs(container: Optional[st.container] = None) -> None:
    """Display logs in the Streamlit UI with error handling."""
    try:
        # Use provided container or st directly
        display = container if container else st

        if 'logs' in st.session_state:
            for log in reversed(st.session_state.logs):
                message = f"{log['timestamp']} - {log['logger_name']} - {log['message']}"

                if log['level'] in ('ERROR', 'CRITICAL'):
                    display.error(message)
                elif log['level'] == 'WARNING':
                    display.warning(message)
                else:
                    display.info(message)

    except Exception as e:
        error_msg = f"Error displaying logs: {str(e)}"
        print(error_msg)
        if container:
            container.error(error_msg)