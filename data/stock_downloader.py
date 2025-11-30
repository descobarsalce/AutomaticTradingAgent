"""Unified stock data downloader with multiple source support."""

import yfinance as yf
import pandas as pd
from datetime import datetime, date
from alpha_vantage.timeseries import TimeSeries
import logging
from typing import Optional
import os
import time
import requests

logger = logging.getLogger(__name__)

class StockDownloader:
    """Unified stock data downloader supporting multiple sources."""

    def __init__(self, source: str = 'yahoo'):
        """Initialize downloader with date range and source."""
        self.source = source.lower()

        if self.source == 'alpha_vantage':
            import streamlit as st
            try:
                self.av_api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
                if not self.av_api_key:
                    raise ValueError("Alpha Vantage API key is empty in secrets.toml")
            except KeyError:
                raise ValueError("Alpha Vantage API key not found in secrets.toml. Please add ALPHA_VANTAGE_API_KEY to .streamlit/secrets.toml")

        logger.info(f"Initialized StockDownloader with source: {source}")

    def download_stock_data(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Download stock data from specified source with unified format."""
        logger.info(f"download_stock_data called: source={self.source}, symbol={symbol}, start={start_date}, end={end_date}")
        if self.source == 'yahoo':
            return self._download_yahoo(symbol, start_date, end_date)
        elif self.source == 'alpha_vantage':
            return self._download_alpha_vantage(symbol, start_date, end_date)
        else:
            raise ValueError(f"Unsupported data source: {self.source}")

    def _download_yahoo(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Download and format data from Yahoo Finance."""
        try:
            logger.info(f"Attempting Yahoo Finance download: symbol={symbol}, start={start_date}, end={end_date}")

            # Convert date to datetime if needed (defensive)
            if isinstance(start_date, date) and not isinstance(start_date, datetime):
                start_date = datetime.combine(start_date, datetime.min.time())
            if isinstance(end_date, date) and not isinstance(end_date, datetime):
                end_date = datetime.combine(end_date, datetime.min.time())

            # Validate date range
            if start_date > end_date:
                raise ValueError(f"Start date ({start_date}) cannot be after end date ({end_date})")

            # Convert to string format that yfinance prefers (YYYY-MM-DD)
            start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
            end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)

            # Log detailed query information
            logger.info(f"=" * 80)
            logger.info(f"YAHOO FINANCE API CALL")
            logger.info(f"=" * 80)
            logger.info(f"yfinance version: {yf.__version__}")
            logger.info(f"Ticker Symbol:    {symbol}")
            logger.info(f"Start Date:       {start_str}")
            logger.info(f"End Date:         {end_str}")
            logger.info(f"Interval:         1d (daily)")
            logger.info(f"API Endpoint:     https://query2.finance.yahoo.com/v8/finance/chart/{symbol}")
            logger.info(f"Query Parameters:")
            logger.info(f"  - period1: {int(start_date.timestamp())}")
            logger.info(f"  - period2: {int(end_date.timestamp())}")
            logger.info(f"  - interval: 1d")
            logger.info(f"  - includeAdjustedClose: true")
            logger.info(f"Equivalent URL:")
            logger.info(f"  https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?period1={int(start_date.timestamp())}&period2={int(end_date.timestamp())}&interval=1d")
            logger.info(f"=" * 80)

            ticker = yf.Ticker(symbol)

            # Check ticker info to see if yfinance can even access this symbol
            try:
                ticker_info = ticker.info
                logger.info(f"Ticker info retrieved: {ticker_info.get('symbol', 'N/A')}, longName: {ticker_info.get('longName', 'N/A')}")
            except Exception as info_error:
                logger.warning(f"Could not retrieve ticker info: {info_error}")

            logger.info(f"Calling ticker.history() with start={start_str}, end={end_str}, interval=1d...")
            df = ticker.history(
                start=start_str,
                end=end_str,
                interval="1d"
            )

            logger.info(f"yfinance returned DataFrame with shape: {df.shape}")
            logger.info(f"DataFrame columns: {list(df.columns)}")
            logger.info(f"DataFrame index type: {type(df.index)}")
            logger.info(f"DataFrame index length: {len(df.index)}")
            if not df.empty:
                logger.info(f"First row date: {df.index[0]}")
                logger.info(f"Last row date: {df.index[-1]}")
                logger.info(f"Sample data (first row): {df.iloc[0].to_dict()}")
            else:
                logger.warning(f"DataFrame is EMPTY - yfinance returned no data")
                # Try to get more info about why
                logger.info(f"Checking if ticker has any historical data...")
                try:
                    # Try different parameters
                    test_df = ticker.history(period="1mo")
                    logger.info(f"Test with period='1mo': {len(test_df)} rows")
                    if not test_df.empty:
                        logger.info(f"1-month data available from {test_df.index[0]} to {test_df.index[-1]}")
                except Exception as test_error:
                    logger.error(f"Test download also failed: {test_error}")

            logger.info(f"Received {len(df)} rows from yfinance for {symbol}")

            if df.empty:
                logger.warning(f"yfinance returned empty DataFrame for {symbol}")
                logger.info(f"Attempting direct Yahoo Finance API fallback...")

                # Try direct API call as fallback
                try:
                    df = self._download_yahoo_direct(symbol, start_date, end_date)
                    if not df.empty:
                        logger.info(f"✓ Direct API fallback successful: {len(df)} rows retrieved")
                        return self._process_data(df)
                except Exception as fallback_error:
                    logger.error(f"Direct API fallback also failed: {fallback_error}")

                return pd.DataFrame()

            result = self._process_data(df)
            logger.info(f"Successfully processed {len(result)} rows for {symbol}")
            return result

        except ValueError as ve:
            logger.error(f"Validation error for {symbol}: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    def _download_yahoo_direct(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Direct download from Yahoo Finance API bypassing yfinance library."""
        try:
            period1 = int(start_date.timestamp())
            period2 = int(end_date.timestamp())

            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'period1': period1,
                'period2': period2,
                'interval': '1d',
                'includeAdjustedClose': 'true'
            }

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            logger.info(f"Direct API call to: {url}")
            logger.info(f"Parameters: {params}")

            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            logger.info(f"API Response status: {response.status_code}")

            # Parse Yahoo Finance API response
            if 'chart' not in data or 'result' not in data['chart']:
                logger.error(f"Unexpected API response format: {data}")
                return pd.DataFrame()

            result = data['chart']['result']
            if not result or len(result) == 0:
                logger.warning(f"No data in API response for {symbol}")
                return pd.DataFrame()

            quotes = result[0]
            timestamps = quotes['timestamp']
            indicators = quotes['indicators']['quote'][0]

            # Create DataFrame
            df = pd.DataFrame({
                'Date': pd.to_datetime(timestamps, unit='s'),
                'Open': indicators['open'],
                'High': indicators['high'],
                'Low': indicators['low'],
                'Close': indicators['close'],
                'Volume': indicators['volume']
            })

            df.set_index('Date', inplace=True)
            df = df.dropna()  # Remove any rows with NaN values

            logger.info(f"Direct API retrieved {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Direct API download failed: {type(e).__name__}: {str(e)}")
            raise

    def _download_alpha_vantage(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Download and format data from Alpha Vantage with retry mechanism."""
        max_retries = 3
        base_delay = 12  # seconds - to respect 5 calls/minute limit
        max_batch_size = 5  # maximum symbols per minute

        try:
            logger.info(f"Attempting Alpha Vantage download for {symbol} with API key: {self.av_api_key[:5]}...")
            ts = TimeSeries(key=self.av_api_key, output_format='pandas')
            # Use 'compact' for free tier (last 100 days), 'full' requires premium
            logger.info(f"Executing: ts.get_daily(symbol={symbol}, outputsize='compact')")
            data, meta = ts.get_daily(symbol=symbol, outputsize='compact')
            logger.info(f"Alpha Vantage response metadata: {meta}")
            logger.info(f"Alpha Vantage returned {len(data)} total rows before date filtering")

            # Filter date range
            data.index = pd.to_datetime(data.index)
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date)
            mask = (data.index >= start_dt) & (data.index <= end_dt)
            df = data[mask]

            if df.empty:
                return pd.DataFrame()

            # Rename columns to match unified format
            df = df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })

            return self._process_data(df)
        except Exception as e:
            logger.error(f"Alpha Vantage error for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate downloaded data."""
        try:
            # Remove duplicates
            df = pd.DataFrame(df[~df.index.duplicated(keep='first')])

            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Forward fill missing values
            df = df[required_columns].ffill()

            # Convert volume to integer
            df['Volume'] = df['Volume'].fillna(0).astype(int)

            # Round prices to 4 decimal places
            price_columns = ['Open', 'High', 'Low', 'Close']
            df[price_columns] = df[price_columns].round(4)

            return df
        except Exception as e:
            logger.error(f"Data processing error: {str(e)}")
            return pd.DataFrame()