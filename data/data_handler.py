import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import and_
from models.database import Session, StockData
from data.data_feature_engineer import FeatureEngineer

class SQLDataManager:
    def __init__(self):
        self.session = Session()

    def get_cached_data(self, symbol, start_date, end_date):
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

            newest_record = max(record.last_updated for record in cached_records)
            if datetime.utcnow() - newest_record > timedelta(days=1):
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
            print(f"Error reading from cache: {e}")
            return None

    def cache_data(self, symbol, data):
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
            print(f"Error caching data: {e}")
            self.session.rollback()


class DataHandler:
    def __init__(self):
        self.portfolio_data = {}
        self.sql_manager = SQLDataManager()
        self.feature_engineer = FeatureEngineer()

    def fetch_data(self, symbols, start_date, end_date):
        if isinstance(symbols, str):
            symbols = [symbols]

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for symbol in symbols:
            try:
                cached_data = self.sql_manager.get_cached_data(symbol, start_date, end_date)
                if cached_data is not None and all(col in cached_data.columns for col in required_columns):
                    self.portfolio_data[symbol] = cached_data
                else:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start_date, end=end_date)
                    if data.empty:
                        raise ValueError(f"No data retrieved for symbol {symbol}")
                    if not all(col in data.columns for col in required_columns):
                        raise ValueError(f"Missing required columns for {symbol}")
                    self.sql_manager.cache_data(symbol, data)
                    self.portfolio_data[symbol] = data
            except Exception as e:
                raise ValueError(f"Error fetching data for {symbol}: {str(e)}")

        return self.portfolio_data

    def prepare_data(self):
        self.portfolio_data = self.feature_engineer.prepare_data(self.portfolio_data)
        return self.portfolio_data