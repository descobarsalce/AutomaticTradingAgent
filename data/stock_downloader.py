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

    def download_stock_data(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        instrument_type: str = "equity",
        expiry: Optional[str] = None,
    ) -> pd.DataFrame:
        """Download market data from the configured source with a unified format.

        Args:
            symbol: The ticker symbol to download.
            start_date: Start of the query window (required for equities).
            end_date: End of the query window (required for equities).
            instrument_type: "equity" for historical OHLCV or "options" for option chains.
            expiry: Optional expiration (YYYY-MM-DD) when requesting options; defaults to all expirations.
        """
        logger.info(
            "download_stock_data called: source=%s, symbol=%s, start=%s, end=%s, instrument_type=%s, expiry=%s",
            self.source,
            symbol,
            start_date,
            end_date,
            instrument_type,
            expiry,
        )

        instrument = instrument_type.lower()

        if instrument == "options":
            if self.source != "yahoo":
                raise ValueError(f"Options download is only supported via Yahoo Finance, not {self.source}")
            return self._download_yahoo_options(symbol, expiry)

        if instrument != "equity":
            raise ValueError(f"Unsupported instrument type: {instrument_type}")

        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date are required for equity downloads")

        if self.source == 'yahoo':
            return self._download_yahoo(symbol, start_date, end_date)
        elif self.source == 'alpha_vantage':
            return self._download_alpha_vantage(symbol, start_date, end_date)
        else:
            raise ValueError(f"Unsupported data source: {self.source}")

    def download_options_data(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        """Convenience wrapper to fetch option chains from Yahoo Finance."""
        logger.info("download_options_data called: source=%s, symbol=%s, expiry=%s", self.source, symbol, expiry)
        if self.source != "yahoo":
            raise ValueError("download_options_data currently supports only the Yahoo source")
        return self._download_yahoo_options(symbol, expiry)

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

    def _download_yahoo_options(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        """Download and normalize option chains from Yahoo Finance.

        Args:
            symbol: Underlying ticker.
            expiry: Optional expiration expressed as YYYY-MM-DD or datetime/date.
        """
        try:
            logger.info(f"Attempting Yahoo Finance options download: symbol={symbol}, expiry={expiry or 'ALL'}")
            logger.info("=" * 80)
            logger.info("YAHOO FINANCE OPTIONS CALL")
            logger.info("=" * 80)
            logger.info(f"yfinance version: {yf.__version__}")
            logger.info(f"Ticker Symbol:    {symbol}")
            logger.info(f"Data Endpoint:    https://query2.finance.yahoo.com/v7/finance/options/{symbol}")
            logger.info(f"Requested Expiry: {expiry or 'ALL AVAILABLE'}")
            logger.info("=" * 80)

            ticker = yf.Ticker(symbol)

            try:
                ticker_info = ticker.info
                logger.info(
                    "Ticker info retrieved: %s, longName: %s",
                    ticker_info.get("symbol", "N/A"),
                    ticker_info.get("longName", "N/A"),
                )
            except Exception as info_error:
                logger.warning(f"Could not retrieve ticker info: {info_error}")

            expirations = list(ticker.options or [])
            logger.info(f"Available expirations ({len(expirations)}): {expirations}")

            if not expirations:
                logger.warning(f"No option expirations available for {symbol}")
                return pd.DataFrame()

            if expiry:
                normalized_expiry = self._normalize_expiry(expiry)
                logger.info(f"Normalized requested expiry to {normalized_expiry}")
                if normalized_expiry not in expirations:
                    raise ValueError(f"Requested expiry {normalized_expiry} not available for {symbol}")
                target_expiries = [normalized_expiry]
            else:
                target_expiries = expirations

            frames = []
            for exp in target_expiries:
                logger.info(f"Fetching option chain for {symbol} expiry {exp}")
                chain = ticker.option_chain(exp)
                logger.info(
                    "option_chain returned calls=%s rows, puts=%s rows",
                    getattr(chain.calls, "shape", [0])[0],
                    getattr(chain.puts, "shape", [0])[0],
                )

                for option_df, option_type in ((chain.calls, "call"), (chain.puts, "put")):
                    if option_df is None or option_df.empty:
                        logger.warning(f"No {option_type} data for expiry {exp}")
                        continue

                    normalized = self._normalize_option_chain(option_df, exp, option_type, symbol)
                    if not normalized.empty:
                        frames.append(normalized)

            if not frames:
                logger.warning(f"No option data retrieved for {symbol}")
                return pd.DataFrame()

            result = pd.concat(frames, ignore_index=True)
            logger.info(
                "Successfully processed %s option rows for %s across %s expirations",
                len(result),
                symbol,
                len(target_expiries),
            )
            return result

        except ValueError as ve:
            logger.error(f"Validation error for {symbol} options: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Yahoo Finance options error for {symbol}: {type(e).__name__}: {str(e)}", exc_info=True)
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

    def _normalize_expiry(self, expiry: date) -> str:
        """Convert an expiry value into the YYYY-MM-DD string format used by yfinance."""
        if isinstance(expiry, str):
            return expiry
        if isinstance(expiry, datetime):
            expiry = expiry.date()
        return expiry.strftime("%Y-%m-%d")

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

    def _normalize_option_chain(
        self, option_df: pd.DataFrame, expiry: str, option_type: str, symbol: str
    ) -> pd.DataFrame:
        """Normalize option chain data into a consistent schema."""
        df = option_df.copy()
        df["Symbol"] = symbol
        df["Expiry"] = pd.to_datetime(self._normalize_expiry(expiry))
        df["Type"] = option_type

        rename_map = {
            "strike": "Strike",
            "bid": "Bid",
            "ask": "Ask",
            "lastPrice": "LastPrice",
            "volume": "Volume",
            "openInterest": "OpenInterest",
        }

        for src, dest in rename_map.items():
            if src not in df.columns:
                df[src] = pd.NA
            df = df.rename(columns={src: dest})

        desired_columns = [
            "Symbol",
            "Expiry",
            "Strike",
            "Type",
            "Bid",
            "Ask",
            "LastPrice",
            "Volume",
            "OpenInterest",
        ]
        df = df.reindex(columns=desired_columns)

        numeric_cols = ["Strike", "Bid", "Ask", "LastPrice", "Volume", "OpenInterest"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Volume"] = df.get("Volume", pd.Series(dtype="float")).fillna(0).astype(int)
        df["OpenInterest"] = df.get("OpenInterest", pd.Series(dtype="float")).fillna(0).astype(int)

        return df[desired_columns]
