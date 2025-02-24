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
        try:
            import os
            self.api_key = os.getenv("alpha_vantage_v2_test")
            if not self.api_key:
                raise ValueError("alpha_vantage_v2_test not found in Replit Secrets")
            self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        except Exception as e:
            logger.error(f"Failed to initialize Alpha Vantage: {e}")
            raise ValueError("Please add alpha_vantage_v2_test to Replit Secrets (Tools > Secrets)")

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