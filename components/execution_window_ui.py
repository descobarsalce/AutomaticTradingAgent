
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import sys

def display_execution_window():
    """Display the code execution interface"""
    st.header("Data Analysis Console")
    with st.expander("Python Code Execution", expanded=True):
        code = st.text_area(
            "Enter Python code:",
            height=300,
            help="Access data via st.session_state.model.data_handler")

        # Initialize persistent namespace in session state if not exists
        if 'code_namespace' not in st.session_state:
            st.session_state.code_namespace = {
                'np': np,
                'pd': pd,
                'plt': plt,
                'go': go,
                'vars': {},  # For user-defined variables
            }

        if st.button("Execute Code"):
            try:
                # Update namespace with latest session state
                st.session_state.code_namespace.update({
                    'data_handler': st.session_state.model.data_handler,
                    'stock_names': st.session_state.stock_names,
                    'train_start_date': st.session_state.train_start_date,
                    'train_end_date': st.session_state.train_end_date,
                    'test_start_date': st.session_state.test_start_date,
                    'test_end_date': st.session_state.test_end_date,
                    'env_params': st.session_state.env_params,
                    'model': st.session_state.model,
                    'vars': st.session_state.code_namespace['vars'],
                })

                # Create reference to vars dict for easier access
                locals().update(st.session_state.code_namespace['vars'])

                # Create string buffer to capture print output
                output_buffer = io.StringIO()
                original_stdout = sys.stdout
                sys.stdout = output_buffer

                # Execute the code and capture output
                with st.spinner("Executing code..."):
                    exec(code, globals(), st.session_state.code_namespace)

                    # Save all newly defined variables
                    st.session_state.code_namespace['vars'].update({
                        k: v
                        for k, v in st.session_state.code_namespace.items()
                        if k not in [
                            'np', 'pd', 'plt', 'go', 'data_handler',
                            'stock_names', 'train_start_date',
                            'train_end_date', 'test_start_date',
                            'test_end_date', 'env_params', 'model', 'vars'
                        ]
                    })

                    # Display any generated plots
                    if 'plt' in st.session_state.code_namespace:
                        st.pyplot(plt.gcf())
                        plt.close()

                    # Get and display the captured output
                    sys.stdout = original_stdout
                    output = output_buffer.getvalue()
                    if output:
                        st.text_area("Output:", value=output, height=250)

            except Exception as e:
                st.error(f"Error executing code: {str(e)}")
            finally:
                # Ensure stdout is restored
                sys.stdout = original_stdout
