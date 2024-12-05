import yfinance as yf
import pandas as pd
import numpy as np

class DataHandler:
    def __init__(self):
        self.portfolio_data = {}
        
    def fetch_data(self, symbols, start_date, end_date):
        """Fetch historical data from Yahoo Finance for multiple symbols"""
        if isinstance(symbols, str):
            symbols = [symbols]
            
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            self.portfolio_data[symbol] = data
        return self.portfolio_data
        
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
