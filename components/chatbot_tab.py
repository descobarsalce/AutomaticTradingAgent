"""
Chatbot Interface Component
Handles the chatbot interface and OpenAI integration for data querying
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import text
from models.database import Session
from openai import OpenAI
import json
import os

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def format_sql_results(df):
    """Format SQL query results for the chatbot"""
    return df.to_string() if not df.empty else "No results found"

def generate_response(user_input: str, sql_result: str = None):
    """Generate response using OpenAI API"""
    system_prompt = """You are a helpful assistant for a trading data analysis platform. 
    You can help users analyze trading data and generate insights. If you're provided with SQL query results,
    analyze them and provide insights. When users request visualizations, suggest the type of plot that would
    be most appropriate for their analysis."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    if sql_result:
        messages.append({"role": "assistant", "content": f"Here's the data you requested:\n{sql_result}"})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using the latest model
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def execute_query(query: str):
    """Execute SQL query and return results as DataFrame"""
    session = Session()
    try:
        result = session.execute(text(query))
        df = pd.DataFrame(result.fetchall())
        if not df.empty:
            df.columns = result.keys()
        return df
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return pd.DataFrame()
    finally:
        session.close()

def display_chatbot_tab():
    """Display the chatbot interface tab"""
    st.title("Trading Data Chatbot")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your trading data..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate and display assistant response
        with st.chat_message("assistant"):
            response_container = st.empty()
            
            # Check if the prompt contains SQL-related keywords
            sql_keywords = ["select", "from", "where", "group by", "order by"]
            is_sql_query = any(keyword in prompt.lower() for keyword in sql_keywords)
            
            if is_sql_query:
                # Execute SQL query
                df = execute_query(prompt)
                if not df.empty:
                    st.dataframe(df)
                    response = generate_response(prompt, format_sql_results(df))
                else:
                    response = "I couldn't execute your SQL query. Please check the syntax or try a different query."
            else:
                response = generate_response(prompt)
            
            response_container.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Display helpful examples
    with st.expander("Example Queries"):
        st.markdown("""
        Try asking questions like:
        - Show me the performance of AAPL stock in the last month
        - What was the highest trading volume for TSLA?
        - SELECT symbol, MAX(close) as max_price FROM stock_data GROUP BY symbol
        - Generate a line plot of daily closing prices
        """)
