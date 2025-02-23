import os
import pandas as pd
import logging
from datetime import datetime
import requests
import time

logger = logging.getLogger(__name__)

class AlphaVantageSource:
    def __init__(self):
        self.api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set")
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch daily data for a given symbol."""
        try:
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "full",
                "apikey": self.api_key
            }

            response = requests.get(self.base_url, params=params)
            if response.status_code != 200:
                logger.error(f"Error fetching data: {response.status_code}")
                return pd.DataFrame()

            data = response.json()
            if "Time Series (Daily)" not in data:
                logger.error("No time series data in response")
                return pd.DataFrame()

            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            df.index = pd.to_datetime(df.index)
            df.columns = ["Open", "High", "Low", "Close", "Volume"]

            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Filter date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            if df.empty:
                logger.warning(f"No data in specified date range for {symbol}")
                return pd.DataFrame()

            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill')

            # Add rate limiting
            time.sleep(12)  # Alpha Vantage free tier allows 5 calls per minute

            return df.round(2)

        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {str(e)}")
            return pd.DataFrame()