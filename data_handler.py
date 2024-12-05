import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import and_
from models import Session, StockData

class DataHandler:
    def __init__(self):
        self.portfolio_data = {}
        self.session = Session()
        
    def fetch_data(self, symbols, start_date, end_date):
        """Fetch historical data from Yahoo Finance for multiple symbols with caching"""
        if isinstance(symbols, str):
            symbols = [symbols]
            
        for symbol in symbols:
            # Check cache first
            cached_data = self._get_cached_data(symbol, start_date, end_date)
            
            if cached_data is not None:
                self.portfolio_data[symbol] = cached_data
            else:
                # Fetch from yfinance if not in cache
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                # Cache the new data
                self._cache_data(symbol, data)
                
                self.portfolio_data[symbol] = data
                
        return self.portfolio_data
        
    def _get_cached_data(self, symbol, start_date, end_date):
        """Check if we have fresh data in the cache"""
        try:
            # Query cached data
            cached_records = self.session.query(StockData).filter(
                and_(
                    StockData.symbol == symbol,
                    StockData.date >= start_date,
                    StockData.date <= end_date
                )
            ).order_by(StockData.date).all()
            
            if not cached_records:
                return None
                
            # Check if data is fresh (less than 1 day old)
            newest_record = max(record.last_updated for record in cached_records)
            if datetime.utcnow() - newest_record > timedelta(days=1):
                return None
                
            # Convert to DataFrame
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
            
    def _cache_data(self, symbol, data):
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
                # Handle conflict resolution by using merge with unique constraint
                existing = self.session.query(StockData).filter(
                    StockData.symbol == symbol,
                    StockData.date == date
                ).first()
                
                if existing:
                    # Update existing record
                    existing.open = row['Open']
                    existing.high = row['High']
                    existing.low = row['Low']
                    existing.close = row['Close']
                    existing.volume = row['Volume']
                    existing.last_updated = datetime.utcnow()
                else:
                    # Insert new record
                    self.session.add(stock_data)
            
            self.session.commit()
            
        except Exception as e:
            print(f"Error caching data: {e}")
            self.session.rollback()
        
    def prepare_data(self):
        """Prepare data for training for all symbols"""
        if not self.portfolio_data:
            raise ValueError("No data available. Please fetch data first.")
            
        prepared_data = {}
        for symbol, data in self.portfolio_data.items():
            # Calculate technical indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['RSI'] = self._calculate_rsi(data['Close'])
            
            # Calculate volatility
            data['Volatility'] = data['Close'].pct_change().rolling(window=20).std()
            
            # Calculate correlation with other stocks
            correlations = {}
            for other_symbol, other_data in self.portfolio_data.items():
                if other_symbol != symbol:
                    correlations[other_symbol] = data['Close'].corr(other_data['Close'])
            data['Correlations'] = str(correlations)
            
            # Remove NaN values
            data = data.dropna()
            prepared_data[symbol] = data
            
        self.portfolio_data = prepared_data
        return self.portfolio_data
        
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
