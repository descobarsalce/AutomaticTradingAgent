
import streamlit as st
import os
import logging

from utils.callbacks import ProgressBarCallback
from environment import SimpleTradingEnv
from core import TradingAgent
import pandas as pd

def main():
    # Initialize session state first
    if 'logs' not in st.session_state:
        st.session_state['logs'] = []
    if 'ppo_params' not in st.session_state:
        st.session_state['ppo_params'] = None

    # Configure logging before creating container
    class StreamlitLogHandler(logging.Handler):
        def emit(self, record):
            log_entry = self.format(record)
            st.session_state.logs.append(log_entry)
            if len(st.session_state.logs) > 100:
                st.session_state.logs = st.session_state.logs[-100:]

    # Set up handler before container creation
    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)

    # Create log container
    log_container = st.sidebar.container()
    log_container.header("Logs")

    # Display logs in sidebar
    with log_container:
        for log in st.session_state.logs:
            st.text(log)
        
    st.title("Trading Agent Configuration")
    
    # Rest of your main function code...
    # (keeping the rest of the file unchanged)

if __name__ == "__main__":
    main()
