
from datetime import datetime, timedelta
from sqlalchemy import and_
import pandas as pd
from .models import Session, StockData

class SQLDataManager:
    def __init__(self):
        self.session = Session()
    
    def get_cached_data(self, symbol: str, start_date, end_date):
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
    
    def cache_data(self, symbol: str, data: pd.DataFrame):
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
