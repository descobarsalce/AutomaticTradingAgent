
"""
Database Explorer Component
Handles database exploration and visualization interface
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import func, distinct
from models.database import StockData
import os

from data.data_handler import DataHandler

def display_database_explorer():
    """Display the database explorer interface"""
    st.title("Database Explorer")
    st.header("Database Statistics")

    # Initialize database session
    data_handler = DataHandler()
    session = data_handler
    
    col1, col2, col3 = st.columns(3)

    # Total unique symbols
    unique_symbols = session.get_session().query(func.count(distinct(
        StockData.symbol))).scalar()
    col1.metric("Total Unique Symbols", unique_symbols)

    # Date range
    min_date = session.get_session().query(func.min(StockData.date)).scalar()
    max_date = session.get_session().query(func.max(StockData.date)).scalar()
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
    symbols = [row[0] for row in session.get_session().query(distinct(StockData.symbol)).all()]

    for symbol in symbols:
        # Get statistics for each stock
        symbol_data = session.get_session().query(
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

    # Query Interface
    st.header("Query Interface")

    # Symbol selection
    symbols = [row[0] for row in session.get_session().query(distinct(StockData.symbol)).all()]
    if symbols:
        selected_symbol = st.selectbox("Select Symbol", symbols)

        # Date range selection
        date_col1, date_col2 = st.columns(2)
        start_date = date_col1.date_input("Start Date",
                                         min_date if min_date else None)
        end_date = date_col2.date_input("End Date",
                                       max_date if max_date else None)

        if st.button("Query Data"):
            # Fetch data
            query_data = session.get_session().query(StockData).filter(
                StockData.symbol == selected_symbol,
                StockData.date >= start_date,
                StockData.date <= end_date).order_by(StockData.date).all()

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
                stats_col1.metric("Average Price",
                                 f"${df['Close'].mean():.2f}")
                stats_col2.metric("Highest Price", f"${df['High'].max():.2f}")
                stats_col3.metric("Lowest Price", f"${df['Low'].min():.2f}")

                # Display table view
                st.subheader("Table View")
                st.dataframe(df)

                # Display chart view
                st.subheader("Chart View")
                fig = go.Figure(data=[
                    go.Candlestick(x=df['Date'],
                                  open=df['Open'],
                                  high=df['High'],
                                  low=df['Low'],
                                  close=df['Close'])
                ])

                fig.update_layout(title=f'{selected_symbol} Price History',
                                 yaxis_title='Price ($)',
                                 template='plotly_dark',
                                 height=600)

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for the selected criteria.")
