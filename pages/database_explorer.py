import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
from sqlalchemy import func, distinct
from models.database import Session
from models.models import StockData

# Page config
st.set_page_config(
    page_title="Database Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database session
session = Session()

# Database Statistics Section
st.title("Database Explorer")
st.header("Database Statistics")

col1, col2, col3 = st.columns(3)

# Total unique symbols
unique_symbols = session.query(func.count(distinct(StockData.symbol))).scalar()
col1.metric("Total Unique Symbols", unique_symbols)

# Date range
min_date = session.query(func.min(StockData.date)).scalar()
max_date = session.query(func.max(StockData.date)).scalar()
date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
col2.metric("Date Range", date_range)

# Database file size
db_size = os.path.getsize('trading_data.db') / (1024 * 1024)  # Convert to MB
col3.metric("Database Size", f"{db_size:.2f} MB")

# Last update timestamps
st.subheader("Last Update Timestamps")
last_updates = session.query(
    StockData.symbol,
    func.max(StockData.last_updated).label('last_update')
).group_by(StockData.symbol).all()

update_df = pd.DataFrame(last_updates, columns=['Symbol', 'Last Update'])
st.dataframe(update_df)

# Query Interface
st.header("Query Interface")

# Symbol selection
symbols = [row[0] for row in session.query(distinct(StockData.symbol)).all()]
selected_symbol = st.selectbox("Select Symbol", symbols)

# Date range selection
date_col1, date_col2 = st.columns(2)
start_date = date_col1.date_input("Start Date", min_date)
end_date = date_col2.date_input("End Date", max_date)

if st.button("Query Data"):
    # Fetch data
    query_data = session.query(StockData).filter(
        StockData.symbol == selected_symbol,
        StockData.date >= start_date,
        StockData.date <= end_date
    ).order_by(StockData.date).all()
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'Date': record.date,
        'Open': record.open,
        'High': record.high,
        'Low': record.low,
        'Close': record.close,
        'Volume': record.volume
    } for record in query_data])
    
    if not df.empty:
        # Calculate basic statistics
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        stats_col1.metric("Average Price", f"${df['Close'].mean():.2f}")
        stats_col2.metric("Highest Price", f"${df['High'].max():.2f}")
        stats_col3.metric("Lowest Price", f"${df['Low'].min():.2f}")
        
        # Display table view
        st.subheader("Table View")
        st.dataframe(df)
        
        # Display chart view
        st.subheader("Chart View")
        fig = go.Figure(data=[go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )])
        
        fig.update_layout(
            title=f'{selected_symbol} Price History',
            yaxis_title='Price ($)',
            template='plotly_dark',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for the selected criteria.")

# Cache Management
st.header("Cache Management")

# Display cached symbols with update buttons
for symbol in symbols:
    col1, col2 = st.columns([3, 1])
    last_update = next((update[1] for update in last_updates if update[0] == symbol), None)
    
    col1.text(f"{symbol} - Last Update: {last_update}")
    if col2.button(f"Refresh {symbol}", key=f"refresh_{symbol}"):
        st.warning(f"Refreshing data for {symbol}...")
        # Note: Actual refresh logic would be implemented in data_handler.py

if st.button("Refresh All Data"):
    st.warning("Refreshing all symbols...")
    # Note: Actual refresh logic would be implemented in data_handler.py
