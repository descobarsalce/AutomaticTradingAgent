
"""Alpha Vantage data source implementation."""

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class AlphaVantageAPI:
    def __init__(self):
        self.api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set")
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        self.request_delay = 12  # Rate limit compliance: 5 calls per minute

    def fetch_intraday_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        try:
            data, _ = self.ts.get_intraday(
                symbol=symbol,
                interval='60min',
                outputsize='full'
            )
            
            if data.empty:
                logger.warning(f"Empty dataset received for {symbol}")
                return None

            # Rename columns to match our format
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Filter date range
            data = data[(data.index >= start_date) & (data.index <= end_date)]
            
            # Handle any missing values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            return data.round(2)

        except Exception as e:
            logger.error(f"Error fetching data from Alpha Vantage for {symbol}: {str(e)}")
            return None
