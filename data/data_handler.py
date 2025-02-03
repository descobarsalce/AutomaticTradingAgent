
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

    def fetch_data(self, symbols, start_date, end_date):
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if isinstance(symbols, str):
            symbols = [symbols]

        all_stocks_data = []
        for symbol in symbols:
            try:
                cached_data = self.get_cached_data(symbol, start_date, end_date)
                if cached_data is not None and all(col in cached_data.columns for col in required_columns):
                    df = cached_data
                else:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date)
                    if df.empty:
                        raise ValueError(f"No data retrieved for {symbol}")
                    if not all(col in df.columns for col in required_columns):
                        raise ValueError(f"Missing required columns for {symbol}")
                    self.cache_data(symbol, df)
                
                # Add suffix to column names
                df.columns = [f"{col}_{symbol}" for col in df.columns]
                all_stocks_data.append(df)
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                raise ValueError(f"Error fetching data for {symbol}: {str(e)}")

        if not all_stocks_data:
            raise ValueError("No valid data retrieved for any symbols")

        # Concatenate all dataframes
        result = pd.concat(all_stocks_data, axis=1)
        return result

    def prepare_data(self, portfolio_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare data using feature engineer"""
        return self.feature_engineer.prepare_data(portfolio_data)

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
                'Open': record.open,
                'High': record.high,
                'Low': record.low,
                'Close': record.close,
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
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    close=row['Close'],
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
            self.session.rollback()
            raise
