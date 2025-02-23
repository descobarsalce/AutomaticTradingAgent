
"""Alpha Vantage data source with enhanced connection pooling and rate limiting."""

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import time
import asyncio
import aiohttp
from typing import Optional, Dict, List
from collections import deque
from functools import wraps

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, calls_per_minute: int = 5):
        self.calls_per_minute = calls_per_minute
        self.calls = deque(maxlen=calls_per_minute)
        
    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            
            # Remove old calls
            while self.calls and now - self.calls[0] >= 60:
                self.calls.popleft()
                
            # If at limit, wait until oldest call is more than 60 seconds old
            if len(self.calls) >= self.calls_per_minute:
                wait_time = 60 - (now - self.calls[0])
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
            
            self.calls.append(now)
            return await func(*args, **kwargs)
        return wrapper

class AlphaVantageAPI:
    def __init__(self):
        self.api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set")
        
        self.base_url = "https://www.alphavantage.co/query"
        self.session = None
        self.rate_limiter = RateLimiter(calls_per_minute=5)

    async def _init_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session

    @RateLimiter(calls_per_minute=5)
    async def _fetch_data(self, symbol: str) -> Optional[Dict]:
        """Fetch intraday data with automatic retries and error handling."""
        session = await self._init_session()
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": "60min",
            "outputsize": "full",
            "apikey": self.api_key
        }
        
        try:
            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Error fetching data: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Request error for {symbol}: {str(e)}")
            return None

    def _process_response(self, data: Dict) -> Optional[pd.DataFrame]:
        """Process API response into DataFrame."""
        if not data or "Time Series (60min)" not in data:
            return None
            
        df = pd.DataFrame.from_dict(data["Time Series (60min)"], orient="index")
        df.index = pd.to_datetime(df.index)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        
        # Convert columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace(r'[^\d.]', ''), errors='coerce')
            
        return df.sort_index()

    async def fetch_intraday_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch and process intraday data."""
        try:
            raw_data = await self._fetch_data(symbol)
            if raw_data is None:
                return None
                
            df = self._process_response(raw_data)
            if df is None:
                return None
                
            # Filter date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            if df.empty:
                logger.warning(f"No data in specified date range for {symbol}")
                return None
                
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df.round(2)
            
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {str(e)}")
            return None

    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
