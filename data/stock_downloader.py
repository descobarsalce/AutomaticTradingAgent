"""Unified stock data downloader with multiple source support."""

import yfinance as yf
import pandas as pd
from datetime import datetime, date
from alpha_vantage.timeseries import TimeSeries
import logging
from typing import Optional, List
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

    def get_options_expirations(self, symbol: str, max_retries: int = 3) -> List[str]:
        """Get available expiration dates for a symbol's options."""
        logger.info(f"=" * 60)
        logger.info(f"FETCHING OPTIONS EXPIRATIONS FOR: {symbol}")
        logger.info(f"=" * 60)

        # Try yfinance first with retries
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}: Creating yf.Ticker for {symbol}...")

                # Create a session with proper headers to avoid blocking
                session = requests.Session()
                session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'application/json,text/html,application/xhtml+xml',
                    'Accept-Language': 'en-US,en;q=0.9',
                })

                ticker = yf.Ticker(symbol, session=session)

                logger.info(f"Accessing ticker.options property...")
                expirations = ticker.options

                logger.info(f"Raw expirations response type: {type(expirations)}")
                logger.info(f"Raw expirations response: {expirations}")

                if expirations:
                    logger.info(f"SUCCESS: Found {len(expirations)} expiration dates for {symbol}")
                    logger.info(f"First expiration: {expirations[0]}")
                    logger.info(f"Last expiration: {expirations[-1]}")
                    return list(expirations)
                else:
                    logger.warning(f"NO EXPIRATIONS: ticker.options returned empty for {symbol}")

            except requests.exceptions.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt + 1} failed with JSONDecodeError: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # exponential backoff: 2s, 4s, 6s
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                continue

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                continue

        # If yfinance fails, try direct API call
        logger.info(f"yfinance failed after {max_retries} attempts, trying direct API...")
        return self._get_options_expirations_direct(symbol)

    def _get_options_expirations_direct(self, symbol: str) -> List[str]:
        """Fallback: Get options expirations directly from Yahoo Finance API."""
        try:
            url = f"https://query2.finance.yahoo.com/v7/finance/options/{symbol}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json',
            }

            logger.info(f"Direct API call to: {url}")
            response = requests.get(url, headers=headers, timeout=10)
            logger.info(f"Response status: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"Direct API returned status {response.status_code}")
                logger.error(f"Response text: {response.text[:500]}")
                return []

            data = response.json()

            if 'optionChain' not in data or 'result' not in data['optionChain']:
                logger.error(f"Unexpected response format: {list(data.keys())}")
                return []

            result = data['optionChain']['result']
            if not result:
                logger.warning(f"No options data in response for {symbol}")
                return []

            expirations = result[0].get('expirationDates', [])

            if expirations:
                # Convert Unix timestamps to date strings
                from datetime import datetime as dt
                exp_dates = [dt.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in expirations]
                logger.info(f"Direct API SUCCESS: Found {len(exp_dates)} expirations")
                return exp_dates
            else:
                logger.warning(f"No expiration dates in response")
                return []

        except Exception as e:
            logger.error(f"Direct API failed: {type(e).__name__}: {e}")
            logger.exception("Full traceback:")
            return []

    def download_options_data(
        self,
        symbol: str,
        expirations: List[str] = None,
        option_type: str = 'both'
    ) -> pd.DataFrame:
        """
        Download options data from Yahoo Finance.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            expirations: List of expiration dates to fetch. If None, fetches all.
            option_type: 'call', 'put', or 'both'

        Returns:
            DataFrame with all options data
        """
        logger.info(f"Downloading options for {symbol}, expirations={expirations}, type={option_type}")

        try:
            # Create session with proper headers
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json,text/html,application/xhtml+xml',
                'Accept-Language': 'en-US,en;q=0.9',
            })

            ticker = yf.Ticker(symbol, session=session)
            available_expirations = ticker.options

            if not available_expirations:
                logger.warning(f"No options available for {symbol}")
                # Try getting expirations via direct API
                available_expirations = self._get_options_expirations_direct(symbol)
                if not available_expirations:
                    return pd.DataFrame()

            # Use all expirations if none specified
            if expirations is None:
                expirations = available_expirations
            else:
                # Validate requested expirations
                expirations = [exp for exp in expirations if exp in available_expirations]
                if not expirations:
                    logger.warning(f"None of the requested expirations are available")
                    return pd.DataFrame()

            all_options = []

            for exp in expirations:
                try:
                    chain = ticker.option_chain(exp)

                    if option_type in ('call', 'both'):
                        calls = chain.calls.copy()
                        calls['option_type'] = 'call'
                        calls['expiration'] = exp
                        calls['ticker'] = symbol
                        all_options.append(calls)

                    if option_type in ('put', 'both'):
                        puts = chain.puts.copy()
                        puts['option_type'] = 'put'
                        puts['expiration'] = exp
                        puts['ticker'] = symbol
                        all_options.append(puts)

                    logger.info(f"Fetched options for {symbol} expiring {exp}")

                except Exception as e:
                    logger.error(f"Failed to fetch {symbol} options for {exp} via yfinance: {e}")
                    # Try direct API fallback for this expiration
                    logger.info(f"Trying direct API for {symbol} {exp}...")
                    direct_data = self._download_option_chain_direct(symbol, exp, option_type)
                    if not direct_data.empty:
                        all_options.append(direct_data)
                    continue

            if not all_options:
                return pd.DataFrame()

            df = pd.concat(all_options, ignore_index=True)

            # Rename columns to match database schema
            column_mapping = {
                'contractSymbol': 'contract_symbol',
                'lastTradeDate': 'last_trade_date',
                'lastPrice': 'last_price',
                'percentChange': 'percent_change',
                'openInterest': 'open_interest',
                'impliedVolatility': 'implied_volatility',
                'inTheMoney': 'in_the_money',
                'contractSize': 'contract_size'
            }
            df = df.rename(columns=column_mapping)

            # Convert in_the_money to string
            if 'in_the_money' in df.columns:
                df['in_the_money'] = df['in_the_money'].astype(str)

            logger.info(f"Downloaded {len(df)} option contracts for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error downloading options for {symbol}: {e}")
            return pd.DataFrame()

    def _download_option_chain_direct(self, symbol: str, expiration: str, option_type: str = 'both') -> pd.DataFrame:
        """Fallback: Download option chain directly from Yahoo Finance API."""
        try:
            # Convert expiration date string to Unix timestamp
            from datetime import datetime as dt
            exp_date = dt.strptime(expiration, '%Y-%m-%d')
            exp_timestamp = int(exp_date.timestamp())

            url = f"https://query2.finance.yahoo.com/v7/finance/options/{symbol}"
            params = {'date': exp_timestamp}
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json',
            }

            logger.info(f"Direct API call: {url}?date={exp_timestamp}")
            response = requests.get(url, params=params, headers=headers, timeout=10)
            logger.info(f"Response status: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"Direct API returned status {response.status_code}")
                return pd.DataFrame()

            data = response.json()

            if 'optionChain' not in data or 'result' not in data['optionChain']:
                logger.error(f"Unexpected response format")
                return pd.DataFrame()

            result = data['optionChain']['result']
            if not result or 'options' not in result[0]:
                logger.warning(f"No options data in response")
                return pd.DataFrame()

            options_data = result[0]['options'][0]
            all_options = []

            # Process calls
            if option_type in ('call', 'both') and 'calls' in options_data:
                calls_df = pd.DataFrame(options_data['calls'])
                calls_df['option_type'] = 'call'
                calls_df['expiration'] = expiration
                calls_df['ticker'] = symbol
                all_options.append(calls_df)
                logger.info(f"Direct API: {len(calls_df)} call contracts")

            # Process puts
            if option_type in ('put', 'both') and 'puts' in options_data:
                puts_df = pd.DataFrame(options_data['puts'])
                puts_df['option_type'] = 'put'
                puts_df['expiration'] = expiration
                puts_df['ticker'] = symbol
                all_options.append(puts_df)
                logger.info(f"Direct API: {len(puts_df)} put contracts")

            if not all_options:
                return pd.DataFrame()

            df = pd.concat(all_options, ignore_index=True)

            # Rename columns to match expected format
            column_mapping = {
                'contractSymbol': 'contract_symbol',
                'lastTradeDate': 'last_trade_date',
                'lastPrice': 'last_price',
                'percentChange': 'percent_change',
                'openInterest': 'open_interest',
                'impliedVolatility': 'implied_volatility',
                'inTheMoney': 'in_the_money',
                'contractSize': 'contract_size'
            }
            df = df.rename(columns=column_mapping)

            # Convert in_the_money to string
            if 'in_the_money' in df.columns:
                df['in_the_money'] = df['in_the_money'].astype(str)

            logger.info(f"Direct API SUCCESS: {len(df)} contracts for {symbol} {expiration}")
            return df

        except Exception as e:
            logger.error(f"Direct option chain download failed: {type(e).__name__}: {e}")
            logger.exception("Full traceback:")
            return pd.DataFrame()