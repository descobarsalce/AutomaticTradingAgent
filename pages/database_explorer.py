import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
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

# Data Management
st.header("Data Management")

# Add new stock symbol
new_symbol = st.text_input("Add New Stock Symbol (e.g., AAPL)", "").upper()
if new_symbol:
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
        
    if st.button("Add Stock"):
        try:
            from data.data_handler import DataHandler
            data_handler = DataHandler()
            
            # Check existing data
            existing_data = data_handler.sql_manager.get_cached_data(new_symbol, start_date, end_date)
            if existing_data is not None:
                earliest_date = existing_data.index.min()
                latest_date = existing_data.index.max()
                st.info(f"Existing data for {new_symbol} from {earliest_date.date()} to {latest_date.date()}")
            
            # Download new data with date range
            st.info(f"Downloading data for {new_symbol} from {start_date} to {end_date}...")
            data = data_handler.fetch_data([new_symbol], start_date, end_date)
            st.success(f"Successfully downloaded data for {new_symbol}")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error downloading data: {str(e)}")

# Display and update existing symbols
st.subheader("Update Existing Data")
for symbol in symbols:
    col1, col2, col3 = st.columns([2, 1, 1])
    last_update = next((update[1] for update in last_updates if update[0] == symbol), None)
    
    col1.text(f"{symbol} - Last Update: {last_update}")
    
    if col2.button(f"Check Gaps {symbol}", key=f"check_{symbol}"):
        from data.data_handler import DataHandler
        data_handler = DataHandler()
        data = data_handler.sql_manager.get_cached_data(symbol, None, None)
        
        if data is not None:
            # Check for gaps in data
            data = data.sort_index()
            date_gaps = data.index.to_series().diff().dt.days > 1
            gaps = date_gaps[date_gaps].index
            
            if len(gaps) > 0:
                st.warning(f"Found {len(gaps)} gaps in data for {symbol}")
                for gap in gaps[:5]:  # Show first 5 gaps
                    st.write(f"Gap around: {gap.date()}")
            else:
                st.success("No gaps found in the data")
    
    if col3.button(f"Update {symbol}", key=f"refresh_{symbol}"):
        try:
            from data.data_handler import DataHandler
            data_handler = DataHandler()
            st.info(f"Updating data for {symbol}...")
            data_handler.fetch_data([symbol], None, None)
            st.success(f"Successfully updated {symbol}")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error updating data: {str(e)}")

if st.button("Update All Symbols"):
    try:
        from data.data_handler import DataHandler
        data_handler = DataHandler()
        progress_bar = st.progress(0)
        for i, symbol in enumerate(symbols):
            st.info(f"Updating {symbol}...")
            data_handler.fetch_data([symbol], None, None)
            progress_bar.progress((i + 1) / len(symbols))
        st.success("Successfully updated all symbols")
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Error updating data: {str(e)}")
