import logging
import streamlit as st
from datetime import datetime
from typing import Optional

class StreamlitLogHandler(logging.Handler):
    """Custom logging handler that writes logs to Streamlit's UI."""
    
    def __init__(self, max_entries: int = 1000):
        super().__init__()
        self.max_entries = max_entries
        if 'log_messages' not in st.session_state:
            st.session_state.log_messages = []
            
    def emit(self, record):
        """Emit a log record to Streamlit's UI."""
        try:
            log_entry = self.format(record)
            
            # Add color coding based on log level
            level_colors = {
                'DEBUG': 'gray',
                'INFO': 'white',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'darkred'
            }
            color = level_colors.get(record.levelname, 'white')
            
            # Create formatted log entry with timestamp and color
            formatted_entry = {
                'timestamp': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
                'level': record.levelname,
                'message': record.getMessage(),
                'color': color
            }
            
            if 'log_messages' in st.session_state:
                st.session_state.log_messages.append(formatted_entry)
                # Keep only the latest entries
                if len(st.session_state.log_messages) > self.max_entries:
                    st.session_state.log_messages = st.session_state.log_messages[-self.max_entries:]
            
            # Also print to console for debugging
            print(f"{formatted_entry['timestamp']} - {record.levelname} - {record.getMessage()}")
            
        except Exception as e:
            print(f"Logging error: {e}")

def setup_logging(level: str = 'INFO', max_entries: int = 1000) -> None:
    """
    Set up logging configuration for the entire application.
    
    Args:
        level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_entries: Maximum number of log entries to keep in memory
    """
    # Create handlers
    streamlit_handler = StreamlitLogHandler(max_entries=max_entries)
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

def display_logs(container: Optional[st.container] = None) -> None:
    """
    Display logs in the Streamlit UI with color coding.
    
    Args:
        container: Optional Streamlit container to display logs in
    """
    if 'log_messages' not in st.session_state:
        return
        
    target = container if container else st
    
    for log in st.session_state.log_messages:
        message = f"{log['timestamp']} - {log['level']} - {log['message']}"
        if log['level'] == 'ERROR' or log['level'] == 'CRITICAL':
            target.error(message)
        elif log['level'] == 'WARNING':
            target.warning(message)
        elif log['level'] == 'INFO':
            target.info(message)
        else:
            target.text(message)
