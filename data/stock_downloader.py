
"""Unified stock data downloader with multiple source support."""

import yfinance as yf
import pandas as pd
from datetime import datetime, date
from alpha_vantage.timeseries import TimeSeries
import logging
from typing import Optional
import os

logger = logging.getLogger(__name__)

class StockDownloader:
    """Unified stock data downloader supporting multiple sources."""
    
    def __init__(self, source: str = 'yahoo'):
        """Initialize downloader with date range and source."""
        self.source = source.lower()
        
        if self.source == 'alpha_vantage':
            self.av_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            if not self.av_api_key:
                raise ValueError("Alpha Vantage API key not found in environment variables")
        
        logger.info(f"Initialized StockDownloader with source: {source}")

    def download_stock_data(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Download stock data from specified source with unified format."""
        try:
            if self.source == 'yahoo':
                return self._download_yahoo(symbol, start_date, end_date)
            elif self.source == 'alpha_vantage':
                return self._download_alpha_vantage(symbol, start_date, end_date)
            else:
                raise ValueError(f"Unsupported data source: {self.source}")
        except Exception as e:
            logger.error(f"Failed to download data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _download_yahoo(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Download and format data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval="1d"
            )
            return self._process_data(df) if not df.empty else pd.DataFrame()
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _download_alpha_vantage(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Download and format data from Alpha Vantage."""
        try:
            logger.info(f"Attempting Alpha Vantage download for {symbol} with API key: {self.av_api_key[:5]}...")
            ts = TimeSeries(key=self.av_api_key, output_format='pandas')
            logger.info(f"Executing: ts.get_daily(symbol={symbol}, outputsize='full')")
            data, meta = ts.get_daily(symbol=symbol, outputsize='full')
            logger.info(f"Alpha Vantage response metadata: {meta}")
            
            # Filter date range
            data.index = pd.to_datetime(data.index)
            mask = (data.index >= start_date) & (data.index <= end_date)
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
