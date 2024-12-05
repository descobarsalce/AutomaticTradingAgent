import yfinance as yf
import pandas as pd
import numpy as np

class DataHandler:
    def __init__(self):
        self.data = None
        
    def fetch_data(self, symbol, start_date, end_date):
        """Fetch historical data from Yahoo Finance"""
        ticker = yf.Ticker(symbol)
        self.data = ticker.history(start=start_date, end=end_date)
        return self.data
        
    def prepare_data(self):
        """Prepare data for training"""
        if self.data is None:
            raise ValueError("No data available. Please fetch data first.")
            
        # Calculate technical indicators
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['RSI'] = self._calculate_rsi(self.data['Close'])
        
        # Remove NaN values
        self.data = self.data.dropna()
        return self.data
        
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
