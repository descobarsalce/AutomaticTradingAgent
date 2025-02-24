"""
Database Explorer Component
Handles database exploration and visualization interface
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import func, distinct
from data.database import StockData
from data.data_handler import AlphaVantageSource, YFinanceSource
from datetime import datetime, timedelta
import os

def display_database_explorer():
    """Display the database explorer interface"""
    st.title("Database Explorer")
    st.header("Database Statistics")

    data_handler = st.session_state.data_handler

    col1, col2, col3 = st.columns(3)

    # Total unique symbols
    unique_symbols = data_handler.query(func.count(distinct(
        StockData.symbol))).scalar()
    col1.metric("Total Unique Symbols", unique_symbols)

    # Date range
    min_date = data_handler.query(func.min(StockData.date)).scalar()
    max_date = data_handler.query(func.max(StockData.date)).scalar()
    if min_date and max_date:
        date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
        col2.metric("Date Range", date_range)

    # Database file size
    if os.path.exists('trading_data.db'):
        db_size = os.path.getsize('trading_data.db') / (1024 * 1024)  # Convert to MB
        col3.metric("Database Size", f"{db_size:.2f} MB")

    # Stock Data Summary
    st.header("Stock Data Summary")

    # Query for stock summary information
    stock_summary = []
    symbols = [row[0] for row in data_handler.query(distinct(StockData.symbol)).all()]

    for symbol in symbols:
        # Get statistics for each stock
        symbol_data = data_handler.query(
            StockData.symbol,
            func.min(StockData.date).label('start_date'),
            func.max(StockData.date).label('end_date'),
            func.count(StockData.id).label('data_points'),
            func.max(StockData.last_updated).label('last_update')).filter(
                StockData.symbol == symbol).group_by(StockData.symbol).first()

        if symbol_data:
            # Calculate coverage percentage
            total_days = (symbol_data.end_date - symbol_data.start_date).days + 1
            coverage = (symbol_data.data_points / total_days) * 100

            stock_summary.append({
                'Symbol': symbol,
                'Start Date': symbol_data.start_date.strftime('%Y-%m-%d'),
                'End Date': symbol_data.end_date.strftime('%Y-%m-%d'),
                'Data Points': symbol_data.data_points,
                'Coverage (%)': f"{coverage:.1f}%",
                'Last Update': symbol_data.last_update.strftime('%Y-%m-%d %H:%M:%S')
            })

    if stock_summary:
        # Convert to DataFrame and display
        summary_df = pd.DataFrame(stock_summary)
        st.dataframe(summary_df,
                    column_config={
                        'Symbol': st.column_config.TextColumn('Symbol', width='small'),
                        'Coverage (%)': st.column_config.TextColumn('Coverage (%)',
                                                                width='small'),
                        'Data Points': st.column_config.NumberColumn('Data Points',
                                                                  format="%d")
                    })

        # Add download button for the summary
        csv = summary_df.to_csv(index=False)
        st.download_button("Download Summary CSV",
                          csv,
                          "stock_data_summary.csv",
                          "text/csv",
                          key='download-csv')
    else:
        st.info("No stock data available in the database.")

    # Data Source Selection
    st.header("Add New Stock Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL")
    
    with col2:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    
    with col3:
        end_date = st.date_input("End Date", datetime.now())
        
    source = st.radio("Select Data Source", ["Yahoo Finance", "Alpha Vantage"])
    
    if st.button("Add Stock Data"):
        try:
            data = data_handler.fetch_data([symbol], start_date, end_date)
            if not data.empty:
                st.success(f"Successfully added data for {symbol}")
            else:
                st.error(f"No data found for {symbol}")
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")

    # Query Interface
    st.header("Query Interface")

    # Symbol input
    selected_symbol = st.text_input("Query Stock Symbol", value="AAPL")

    # Date range selection
    date_col1, date_col2 = st.columns(2)
    start_date = date_col1.date_input("Start Date",
                                     min_date if min_date else None)
    end_date = date_col2.date_input("End Date",
                                   max_date if max_date else None)

    if st.button("Query Data"):
        try:
            # Fetch data using DataHandler
            df = data_handler.fetch_data(selected_symbol, start_date, end_date)

            if not df.empty:
                # Calculate basic statistics
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                stats_col1.metric("Average Price",
                                 f"${df[f'Close_{selected_symbol}'].mean():.2f}")
                stats_col2.metric("Highest Price", f"${df[f'High_{selected_symbol}'].max():.2f}")
                stats_col3.metric("Lowest Price", f"${df[f'Low_{selected_symbol}'].min():.2f}")

                # Display table view
                st.subheader("Table View")
                st.dataframe(df)

                # Display chart view
                st.subheader("Chart View")
                fig = go.Figure(data=[
                    go.Candlestick(x=df.index,
                                  open=df[f'Open_{selected_symbol}'],
                                  high=df[f'High_{selected_symbol}'],
                                  low=df[f'Low_{selected_symbol}'],
                                  close=df[f'Close_{selected_symbol}'])
                ])

                fig.update_layout(title=f'{selected_symbol} Price History',
                                 yaxis_title='Price ($)',
                                 template='plotly_dark',
                                 height=600)

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for the selected criteria.")
        except Exception as e:
            st.error(f"Error fetching data: {e}")