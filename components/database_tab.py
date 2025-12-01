"""
Database Explorer Component
Handles database exploration and visualization interface
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import func, distinct
from data.database import StockData, OptionsData
from datetime import datetime, timedelta
import os
import logging

logger = logging.getLogger(__name__)

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
        symbol = st.text_input("Enter Stock Symbol(s)",
                               value="AAPL",
                               help="Enter one or more symbols separated by commas (e.g., AAPL, MSFT, GOOGL)")
    
    with col2:
        start_date = datetime.combine(
            st.date_input("Start Date",
                          value=datetime.now() - timedelta(days=365)),
            datetime.min.time()
        )

    with col3:
        end_date = datetime.combine(
            st.date_input("End Date",
                          value=datetime.now()),
            datetime.min.time()
        )
        
    source = st.radio("Select Data Source", ["Yahoo Finance", "Alpha Vantage"])
    
    if st.button("Add Stock Data"):
        try:
            if not symbol.strip():
                st.error("Please enter a stock symbol")
            elif start_date > end_date:
                st.error("Start date cannot be after end date")
            else:
                # Create detailed diagnostic info box
                with st.expander("📊 Download Details", expanded=True):
                    st.markdown("### Query Parameters")

                    # Show different details based on source
                    if source == "Yahoo Finance":
                        import yfinance as yf
                        start_str = start_date.strftime('%Y-%m-%d')
                        end_str = end_date.strftime('%Y-%m-%d')
                        period1 = int(start_date.timestamp())
                        period2 = int(end_date.timestamp())

                        # Parse symbols
                        symbol_list = [s.strip() for s in symbol.split(',') if s.strip()]

                        info_text = f"""
Symbol(s):        {symbol}
Source:           {source}
yfinance version: {yf.__version__}

DATE PARAMETERS:
Start Date:       {start_str} (timestamp: {period1})
End Date:         {end_str} (timestamp: {period2})
Date Range:       {(end_date - start_date).days} days
Interval:         1d (daily)
"""

                        if len(symbol_list) > 1:
                            info_text += f"\nWill download {len(symbol_list)} symbols: {', '.join(symbol_list)}\n"
                            info_text += "\nAPI ENDPOINTS (one per symbol):\n"
                            for sym in symbol_list:
                                info_text += f"https://query2.finance.yahoo.com/v8/finance/chart/{sym}\n"
                        else:
                            info_text += f"""
API ENDPOINT:
https://query2.finance.yahoo.com/v8/finance/chart/{symbol_list[0]}

FULL API URL:
https://query2.finance.yahoo.com/v8/finance/chart/{symbol_list[0]}?period1={period1}&period2={period2}&interval=1d&includeAdjustedClose=true
"""

                        info_text += f"""
PYTHON CALL:
for symbol in {symbol_list}:
    ticker = yf.Ticker(symbol)
    df = ticker.history(start='{start_str}', end='{end_str}', interval='1d')
"""

                        st.code(info_text, language="text")
                    else:  # Alpha Vantage
                        st.code(f"""
Symbol:      {symbol}
Source:      {source}
Start Date:  {start_date.date()} ({start_date})
End Date:    {end_date.date()} ({end_date})
Date Range:  {(end_date - start_date).days} days

API ENDPOINT:
https://www.alphavantage.co/query

PARAMETERS:
function:    TIME_SERIES_DAILY
symbol:      {symbol}
outputsize:  compact (last 100 days - free tier)
apikey:      [REDACTED]
                        """, language="text")

                    status_placeholder = st.empty()
                    details_placeholder = st.empty()

                logger.info(f"=== STARTING DOWNLOAD: {symbol} from {source} ({start_date.date()} to {end_date.date()}) ===")

                with st.spinner(f"Downloading {symbol} data from {source}..."):
                    status_placeholder.info("🔄 Initiating download request...")
                    # Pass symbol as string to allow comma-separated values
                    data = data_handler.fetch_data(symbol, start_date, end_date, source, use_SQL=False)

                logger.info(f"=== DOWNLOAD COMPLETE: Received {len(data) if data is not None and not data.empty else 0} rows ===")

                # Show detailed results
                with st.expander("📊 Download Details", expanded=True):
                    if not data.empty:
                        # Parse symbols to show in message
                        symbol_list = [s.strip() for s in symbol.split(',') if s.strip()]
                        symbols_str = ', '.join(symbol_list) if len(symbol_list) > 1 else symbol_list[0]

                        st.markdown("### ✅ Download Results")
                        st.code(f"""
Status:         SUCCESS
Symbols:        {symbols_str}
Rows Retrieved: {len(data)}
Columns:        {', '.join(data.columns.tolist())}
Date Range:     {data.index.min()} to {data.index.max()}
Data Shape:     {data.shape}
Memory Usage:   {data.memory_usage(deep=True).sum() / 1024:.2f} KB
                        """, language="text")

                        st.markdown("### 📈 Data Preview")
                        st.dataframe(data.head(10))

                        if len(symbol_list) > 1:
                            st.success(f"✅ Successfully added {len(data)} rows of data for {len(symbol_list)} symbols: {symbols_str}")
                        else:
                            st.success(f"✅ Successfully added {len(data)} rows of data for {symbols_str}")
                    else:
                        st.markdown("### ⚠️ No Data Retrieved")
                        st.code(f"""
Status:        FAILED - No data returned
Symbol(s):     {symbol}
Source:        {source}
Date Range:    {start_date.date()} to {end_date.date()}

Possible Reasons:
1. Symbol '{symbol}' may not exist or is delisted
2. No trading data available for this date range
3. Data source API might be experiencing issues
4. Date range might be outside available historical data
5. Free tier limitations (for Alpha Vantage)

What to try:
- Check if the symbol is correct (e.g., 'AAPL' not 'Apple')
- Try a different date range (e.g., last 30 days)
- Try the other data source (Yahoo Finance vs Alpha Vantage)
- Check your internet connection
- Look at the console logs for more technical details
                        """, language="text")
                        st.warning(f"⚠️ No data found for {symbol}")

        except ValueError as ve:
            st.error(f"❌ Invalid input: {str(ve)}")
            with st.expander("🔍 Error Details", expanded=True):
                st.code(f"""
Error Type: ValueError
Message:    {str(ve)}
Symbol:     {symbol}
Source:     {source}
Dates:      {start_date.date()} to {end_date.date()}
                """, language="text")

        except Exception as e:
            logger.exception(f"Error fetching data for {symbol}")
            st.error(f"❌ Download Failed: {type(e).__name__}")

            with st.expander("🔍 Error Details", expanded=True):
                st.code(f"""
Error Type:  {type(e).__name__}
Error Message: {str(e)}

Query Parameters:
Symbol:      {symbol}
Source:      {source}
Start Date:  {start_date.date()}
End Date:    {end_date.date()}

Troubleshooting:
1. Check the console/terminal logs for full stack trace
2. Verify your internet connection
3. Try a different symbol or date range
4. For Alpha Vantage: Check your API key is valid
5. For Yahoo Finance: Try again in a few minutes (rate limiting)

Full Error:
{str(e)}
                """, language="text")

                if hasattr(e, '__traceback__'):
                    import traceback
                    st.markdown("### Stack Trace")
                    st.code(traceback.format_exc(), language="python")

    # Options Data Section
    st.header("Download Options Data")

    opt_col1, opt_col2 = st.columns(2)

    with opt_col1:
        options_symbol = st.text_input(
            "Options Symbol",
            value="AAPL",
            key="options_symbol",
            help="Enter a stock ticker to fetch its options chain"
        )

    with opt_col2:
        option_type = st.selectbox(
            "Option Type",
            options=["Both", "Calls Only", "Puts Only"],
            key="option_type"
        )

    # Fetch and display available expirations
    if st.button("Load Available Expirations", key="load_expirations"):
        import yfinance as yf

        with st.expander("Options Query Details", expanded=True):
            st.markdown("### Query Parameters")
            st.code(f"""
Symbol:           {options_symbol}
yfinance version: {yf.__version__}

API Call:
ticker = yf.Ticker("{options_symbol}")
expirations = ticker.options
            """, language="text")

            status_placeholder = st.empty()
            details_placeholder = st.empty()

        try:
            status_placeholder.info("Fetching options expirations (with retry support)...")
            details_placeholder.info("Will retry up to 3 times with exponential backoff, then fall back to direct API...")

            from data.stock_downloader import StockDownloader
            downloader = StockDownloader(source='yahoo')
            expirations = downloader.get_options_expirations(options_symbol)

            if expirations:
                st.session_state['available_expirations'] = list(expirations)
                st.session_state['options_symbol_loaded'] = options_symbol
                status_placeholder.success(f"Found {len(expirations)} expiration dates for {options_symbol}")
                details_placeholder.code(f"""
Status:      SUCCESS
Expirations: {len(expirations)} dates available
First:       {expirations[0] if expirations else 'N/A'}
Last:        {expirations[-1] if expirations else 'N/A'}

All Expirations:
{chr(10).join(expirations[:10])}{'...' if len(expirations) > 10 else ''}
                """, language="text")
            else:
                status_placeholder.warning(f"No options data returned for {options_symbol}")
                details_placeholder.code(f"""
Status:      NO DATA
Symbol:      {options_symbol}

Possible Reasons:
1. The symbol '{options_symbol}' may not have listed options
2. The symbol might be incorrect or delisted
3. Yahoo Finance API may be experiencing issues
4. Some ETFs and stocks don't have options markets

What to try:
- Verify the symbol is correct (case-sensitive)
- Try a major stock like AAPL, MSFT, TSLA, SPY
- Check if the symbol has options on finance.yahoo.com
- Try again in a few minutes (rate limiting)
                """, language="text")

        except Exception as e:
            logger.exception(f"Error fetching options expirations for {options_symbol}")
            status_placeholder.error(f"Failed to fetch expirations: {type(e).__name__}")
            details_placeholder.code(f"""
Status:        FAILED
Error Type:    {type(e).__name__}
Error Message: {str(e)}
Symbol:        {options_symbol}

Troubleshooting:
1. Check your internet connection
2. Verify the symbol exists on Yahoo Finance
3. Try again in a few minutes (API rate limiting)
4. Check console logs for full stack trace

Full Error:
{str(e)}
            """, language="text")

            if hasattr(e, '__traceback__'):
                import traceback
                st.markdown("### Stack Trace")
                st.code(traceback.format_exc(), language="python")

    # Show expiration selector if expirations were loaded
    if 'available_expirations' in st.session_state and st.session_state['available_expirations']:
        expirations = st.session_state['available_expirations']

        exp_selection = st.radio(
            "Expiration Selection",
            options=["All Expirations", "Select Specific"],
            key="exp_selection"
        )

        selected_expirations = None
        if exp_selection == "Select Specific":
            selected_expirations = st.multiselect(
                "Select Expirations",
                options=expirations,
                default=[expirations[0]] if expirations else [],
                key="selected_expirations"
            )

        if st.button("Download Options Data", key="download_options"):
            import yfinance as yf

            # Map option type selection
            type_map = {
                "Both": "both",
                "Calls Only": "call",
                "Puts Only": "put"
            }
            opt_type = type_map[option_type]
            exps_to_fetch = selected_expirations if selected_expirations else expirations

            with st.expander("Options Download Details", expanded=True):
                st.markdown("### Download Parameters")
                st.code(f"""
Symbol:           {options_symbol}
Option Type:      {option_type} ({opt_type})
yfinance version: {yf.__version__}
Expirations:      {len(exps_to_fetch)} dates selected

API Calls:
ticker = yf.Ticker("{options_symbol}")
for exp in expirations:
    chain = ticker.option_chain(exp)
    calls = chain.calls  # if type includes calls
    puts = chain.puts    # if type includes puts
                """, language="text")

                download_status = st.empty()
                download_details = st.empty()

            try:
                download_status.info(f"Downloading options for {options_symbol}...")

                from data.stock_downloader import StockDownloader
                downloader = StockDownloader(source='yahoo')
                options_df = downloader.download_options_data(
                    symbol=options_symbol,
                    expirations=selected_expirations,
                    option_type=opt_type
                )

                if not options_df.empty:
                    download_status.success(f"Downloaded {len(options_df)} option contracts")
                    download_details.code(f"""
Status:           SUCCESS
Total Contracts:  {len(options_df)}
Calls:            {len(options_df[options_df['option_type'] == 'call']) if 'option_type' in options_df.columns else 'N/A'}
Puts:             {len(options_df[options_df['option_type'] == 'put']) if 'option_type' in options_df.columns else 'N/A'}
Expirations:      {options_df['expiration'].nunique() if 'expiration' in options_df.columns else 'N/A'}
Strike Range:     ${options_df['strike'].min():.2f} - ${options_df['strike'].max():.2f}
Columns:          {', '.join(options_df.columns.tolist())}
                    """, language="text")

                    with st.expander("Options Data Preview", expanded=True):
                        # Show summary stats
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Contracts", len(options_df))
                        col2.metric("Expirations", options_df['expiration'].nunique())
                        col3.metric("Strike Range", f"${options_df['strike'].min():.0f} - ${options_df['strike'].max():.0f}")

                        # Display data
                        display_cols = ['contract_symbol', 'option_type', 'expiration', 'strike',
                                       'last_price', 'bid', 'ask', 'volume', 'open_interest', 'implied_volatility']
                        display_cols = [c for c in display_cols if c in options_df.columns]
                        st.dataframe(options_df[display_cols].head(50))

                    # Save to database option
                    if st.button("Save to Database", key="save_options"):
                        try:
                            from sqlalchemy.orm import Session
                            from utils.db_config import db_config

                            with Session(db_config.engine) as session:
                                for _, row in options_df.iterrows():
                                    option = OptionsData(
                                        ticker=row.get('ticker'),
                                        contract_symbol=row.get('contract_symbol'),
                                        option_type=row.get('option_type'),
                                        expiration=pd.to_datetime(row.get('expiration')),
                                        strike=row.get('strike'),
                                        last_price=row.get('last_price'),
                                        bid=row.get('bid'),
                                        ask=row.get('ask'),
                                        change=row.get('change'),
                                        percent_change=row.get('percent_change'),
                                        volume=row.get('volume'),
                                        open_interest=row.get('open_interest'),
                                        implied_volatility=row.get('implied_volatility'),
                                        in_the_money=str(row.get('in_the_money')),
                                        contract_size=row.get('contract_size'),
                                        currency=row.get('currency'),
                                        last_trade_date=row.get('last_trade_date')
                                    )
                                    session.merge(option)
                                session.commit()
                            st.success(f"Saved {len(options_df)} contracts to database")
                        except Exception as e:
                            st.error(f"Error saving to database: {e}")

                    # Download CSV option
                    csv = options_df.to_csv(index=False)
                    st.download_button(
                        "Download as CSV",
                        csv,
                        f"{options_symbol}_options.csv",
                        "text/csv",
                        key='download-options-csv'
                    )
                else:
                    download_status.warning(f"No options data returned for {options_symbol}")
                    download_details.code(f"""
Status:        NO DATA RETURNED
Symbol:        {options_symbol}
Option Type:   {option_type}
Expirations:   {len(exps_to_fetch)} requested

Possible Reasons:
1. Yahoo Finance API returned empty data
2. The requested expirations may have no contracts
3. API rate limiting may be affecting the response
4. Network issues during download

What to try:
- Try fewer expirations (start with just one)
- Try a different symbol (AAPL, SPY, MSFT)
- Wait a few minutes and retry
- Check the console logs for more details
                    """, language="text")

            except Exception as e:
                logger.exception(f"Error downloading options for {options_symbol}")
                download_status.error(f"Download Failed: {type(e).__name__}")
                download_details.code(f"""
Status:        FAILED
Error Type:    {type(e).__name__}
Error Message: {str(e)}
Symbol:        {options_symbol}
Option Type:   {option_type}
Expirations:   {len(exps_to_fetch)} requested

Troubleshooting:
1. Check your internet connection
2. Yahoo Finance may be rate limiting requests
3. Try again in a few minutes
4. Try with fewer expirations
5. Check console logs for full stack trace

Full Error:
{str(e)}
                """, language="text")

                if hasattr(e, '__traceback__'):
                    import traceback
                    st.markdown("### Stack Trace")
                    st.code(traceback.format_exc(), language="python")

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
            df = data_handler._sql_handler.get_cached_data(selected_symbol, start_date, end_date)

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