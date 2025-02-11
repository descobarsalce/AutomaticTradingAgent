"""
Data handler with centralized database configuration and improved error handling.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sqlalchemy import and_, distinct
from datetime import timedelta, datetime
from data.data_feature_engineer import FeatureEngineer
from utils.db_config import get_db_session
import logging
from typing import Dict, Optional
from models.database import StockData

logger = logging.getLogger(__name__)

class DataHandler:

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.session = None

    def get_session(self):
        """Get a database session using context management"""
        if self.session is None or not self.session.is_active:
            try:
                self.session = next(get_db_session())
            except Exception as e:
                logger.error(f"Failed to create database session: {str(e)}")
                raise
        return self.session

    def prepare_data(self, data):
        """Prepare data using the feature engineer"""
        return self.feature_engineer.prepare_data(data)

    def prepare_data_with_extra_features(self, data):
        """Prepare data with extra features using the feature engineer"""
        prepared_data = self.feature_engineer.prepare_data(data)
        symbols = sorted(list(set(col.split('_')[1] for col in prepared_data.columns if '_' in col)))

        for symbol in symbols:
            close_col = f'Close_{symbol}'
            if close_col in prepared_data.columns:
                # Add lagged features
                prepared_data = self.feature_engineer.add_lagged_features(prepared_data, [close_col], lags=5)
                # Add rolling features
                prepared_data = self.feature_engineer.add_rolling_features(prepared_data, close_col, [7, 14, 30])
                # Add Fourier transform features
                fft_features = self.feature_engineer.add_fourier_transform(prepared_data[close_col])
                if not fft_features.empty:
                    for col in fft_features.columns:
                        prepared_data[f'{col}_{symbol}'] = fft_features[col]

        return prepared_data

    def _fetch_cached_data_if_valid(self, symbol: str, start_date, end_date) -> Optional[pd.DataFrame]:
        """Return valid cached data if present and complete, else None."""
        cached_data = self.get_cached_data(symbol, start_date, end_date)
        if cached_data is not None:
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            date_range = pd.date_range(start=start_date, end=end_date, freq='B')
            if all(col in cached_data.columns for col in required_columns) \
                    and len(date_range.difference(cached_data.index)) == 0:
                return cached_data
        return None

    def _fetch_data_yf(self, symbol: str, start_date, end_date) -> pd.DataFrame:
        """Fetch data from yfinance and validate columns."""
        required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"No data retrieved for {symbol}")
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns for {symbol}")
        return df

    def fetch_data(self, symbols, start_date, end_date):
        """Fetch data either from cache or yfinance, suffix columns by symbol."""
        if isinstance(symbols, str):
            symbols = [symbols]

        all_stocks_data = pd.DataFrame()
        
        for symbol in symbols:
            try:
                cached_data = self._fetch_cached_data_if_valid(symbol, start_date, end_date)
                if cached_data is not None:
                    stock_data = cached_data
                else:
                    stock_data = self._fetch_data_yf(symbol, start_date, end_date)
                    self.cache_data(symbol, stock_data)
                
                # Rename columns to include symbol
                stock_data.columns = [f'{col}_{symbol}' for col in stock_data.columns]
                all_stocks_data = pd.concat([all_stocks_data, stock_data], axis=1)
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        if all_stocks_data.empty:
            raise ValueError("No valid data retrieved for any symbols")
            
        return all_stocks_data

    def __del__(self):
        """Cleanup database session"""
        if self.session is not None:
            self.session.close()

    def get_cached_data(self, symbol: str, start_date, end_date) -> Optional[pd.DataFrame]:
        """Check if we have fresh data in the cache and validate continuity"""
        try:
            cached_records = self.session.query(StockData).filter(
                and_(
                    StockData.symbol == symbol,
                    StockData.date >= start_date,
                    StockData.date <= end_date
                )
            ).order_by(StockData.date).all()

            if not cached_records:
                return None

            # Check data continuity
            dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
            available_dates = set(record.date.date() for record in cached_records)
            missing_dates = [d for d in dates if d.date() not in available_dates]
            
            if len(missing_dates) > 0:
                logger.warning(f"Missing {len(missing_dates)} trading days for {symbol}")
                return None  # Force refetch if data is incomplete
                
            newest_record = max(record.last_updated for record in cached_records)
            if datetime.now(datetime.timezone.utc) - newest_record > timedelta(days=1):
                return None

            data = pd.DataFrame([{
                'Close': record.close,
                'Open': record.open,
                'High': record.high,
                'Low': record.low,
                'Volume': record.volume,
                'Date': record.date
            } for record in cached_records])

            data.set_index('Date', inplace=True)
            return data

        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            return None

    def cache_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Cache the fetched data in the database"""
        try:
            for date, row in data.iterrows():
                stock_data = StockData(
                    symbol=symbol,
                    date=date,
                    close=row['Close'],
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    volume=row['Volume'],
                    last_updated=datetime.utcnow()
                )

                existing = self.session.query(StockData).filter(
                    StockData.symbol == symbol,
                    StockData.date == date
                ).first()

                if existing:
                    existing.open = row['Open']
                    existing.high = row['High']
                    existing.low = row['Low']
                    existing.close = row['Close']
                    existing.volume = row['Volume']
                    existing.last_updated = datetime.utcnow()
                else:
                    self.session.add(stock_data)

            self.session.commit()

        except Exception as e:
            logger.error(f"Error caching data: {e}")
            if self.session.is_active:
                self.session.rollback()
            raise
