import logging
import pandas as pd
from datetime import datetime
import streamlit as st
from alpha_vantage.timeseries import TimeSeries
from data.base_sources import DataSource

logger = logging.getLogger(__name__)

class AlphaVantageSource(DataSource):
    """Alpha Vantage implementation of data source."""
    def __init__(self):
        # Try getting key from Replit secrets first
        try:
            import os
            self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        except:
            self.api_key = None
            
        # Fallback to Streamlit secrets if not found
        if not self.api_key:
            try:
                self.api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
            except:
                self.api_key = None
                
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not found in secrets")
            
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')

    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            data, _ = self.ts.get_daily(symbol=symbol, outputsize='full')
            if data.empty:
                raise ValueError(f"Empty dataset received for {symbol}")

            # Rename columns to match format
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

            # Filter date range
            data = data[(data.index >= start_date) & (data.index <= end_date)]

            if data.empty:
                raise ValueError(f"No data in specified date range for {symbol}")

            return data.round(2)

        except Exception as e:
            logger.warning(f"Alpha Vantage error for {symbol}: {str(e)}")
            return pd.DataFrame()