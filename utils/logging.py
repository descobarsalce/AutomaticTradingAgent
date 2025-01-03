import logging
import streamlit as st
from datetime import datetime
from typing import Optional, Dict, Any, Union

class StreamlitLogHandler(logging.Handler):
    """Custom logging handler that writes logs to Streamlit's UI."""

    def __init__(self, max_entries: int = 1000):
        super().__init__()
        self.max_entries = max_entries
        if 'log_messages' not in st.session_state:
            st.session_state.log_messages = []

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to Streamlit's UI.

        Args:
            record: The log record to emit
        """
        try:
            # Add color coding based on log level
            level_colors = {
                'DEBUG': 'gray',
                'INFO': 'white',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'darkred'
            }

            # Create formatted log entry
            formatted_entry: Dict[str, Any] = {
                'timestamp': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'level': record.levelname,
                'message': self.format(record),
                'color': level_colors.get(record.levelname, 'white')
            }

            # Append to session state
            if 'log_messages' in st.session_state:
                st.session_state.log_messages.append(formatted_entry)
                # Keep only the latest entries
                if len(st.session_state.log_messages) > self.max_entries:
                    st.session_state.log_messages = st.session_state.log_messages[-self.max_entries:]

            # Also print to console for debugging
            print(f"{formatted_entry['timestamp']} - {formatted_entry['level']} - {formatted_entry['message']}")

        except Exception as e:
            print(f"Error in StreamlitLogHandler.emit: {str(e)}")

def setup_logging(level: str = 'INFO', max_entries: int = 1000) -> None:
    """
    Set up logging configuration for the entire application.

    Args:
        level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_entries: Maximum number of log entries to keep in memory
    """
    try:
        # Create handlers
        streamlit_handler = StreamlitLogHandler(max_entries=max_entries)
        console_handler = logging.StreamHandler()

        # Create formatter
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
        streamlit_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Get root logger
        root_logger = logging.getLogger()

        # Remove existing handlers to avoid duplicates
        root_logger.handlers = []

        # Add handlers
        root_logger.addHandler(streamlit_handler)
        root_logger.addHandler(console_handler)

        # Set level
        root_logger.setLevel(level.upper())

        # Log successful setup
        root_logger.debug("Logging system initialized successfully")

    except Exception as e:
        print(f"Error in setup_logging: {str(e)}")
        raise

def display_logs(container: Optional[st.container] = None) -> None:
    """
    Display logs in the Streamlit UI with color coding.

    Args:
        container: Optional Streamlit container to display logs in
    """
    if not st.session_state.get('log_messages'):
        if container:
            container.info("No logs available yet")
        return

    target = container if container else st

    try:
        # Get selected log level from session state or sidebar
        log_level = st.session_state.get('selected_log_level', 'ALL')

        # Filter logs based on selected level
        logs_to_display = st.session_state.log_messages
        if log_level != 'ALL':
            logs_to_display = [
                log for log in logs_to_display 
                if isinstance(log, dict) and log.get('level') == log_level
            ]

        # Show newest logs first
        for log in reversed(logs_to_display):
            if not isinstance(log, dict):
                continue

            try:
                # Extract log components with safe fallbacks
                timestamp = log.get('timestamp', '')
                level = log.get('level', 'INFO')
                message = log.get('message', 'No message available')

                # Format the log message
                formatted_message = f"{timestamp} - {level} - {message}"

                # Display with appropriate styling
                if level in ('ERROR', 'CRITICAL'):
                    target.error(formatted_message)
                elif level == 'WARNING':
                    target.warning(formatted_message)
                elif level == 'INFO':
                    target.info(formatted_message)
                else:
                    target.text(formatted_message)

            except Exception as e:
                print(f"Error formatting log entry: {str(e)}")
                continue

    except Exception as e:
        print(f"Error in display_logs: {str(e)}")