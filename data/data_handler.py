import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.data_feature_engineer import FeatureEngineer
from data.sql_data_manager import SQLDataManager

class DataHandler:

    def __init__(self):
        self.portfolio_data = {}
        self.sql_manager = SQLDataManager()
        self.feature_engineer = FeatureEngineer()

    def fetch_data(self, symbols, start_date, end_date):
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if isinstance(symbols, str):
            symbols = [symbols]

        for symbol in symbols:
            try:
                cached_data = self.sql_manager.get_cached_data(
                    symbol, start_date, end_date)
                if cached_data is not None and all(
                        col in cached_data.columns
                        for col in required_columns):
                    self.portfolio_data[symbol] = cached_data
                else:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start_date, end=end_date)
                    if data.empty:
                        raise ValueError(f"No data retrieved for {symbol}")
                    if not all(col in data.columns
                               for col in required_columns):
                        raise ValueError(
                            f"Missing required columns for {symbol}")
                    self.sql_manager.cache_data(symbol, data)
                    self.portfolio_data[symbol] = data
            except Exception as e:
                raise ValueError(f"Error fetching data for {symbol}: {str(e)}")

        if not self.portfolio_data:
            raise ValueError("No valid data retrieved for any symbols")

        return self.portfolio_data

    # This function operates directly on the portfolio_data dictionary saved in the class.
    def prepare_data(self):
        self.portfolio_data = self.feature_engineer.prepare_data(
            self.portfolio_data)
        return self.portfolio_data
