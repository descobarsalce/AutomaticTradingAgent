
from typing import Dict, Optional
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from models import Session, StockData
from data.processing import FeatureEngineer

class DataHandler:
    def __init__(self):
        self.portfolio_data = {}
        self.sql_manager = SQLDataManager()
        self.feature_engineer = FeatureEngineer()

    def fetch_data(self, symbols, start_date, end_date):
        if isinstance(symbols, str):
            symbols = [symbols]

        for symbol in symbols:
            cached_data = self.sql_manager.get_cached_data(symbol, start_date, end_date)
            if cached_data is not None:
                self.portfolio_data[symbol] = cached_data
            else:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                self.sql_manager.cache_data(symbol, data)
                self.portfolio_data[symbol] = data

        return self.portfolio_data

    def prepare_data(self):
        self.portfolio_data = self.feature_engineer.prepare_data(self.portfolio_data)
        return self.portfolio_data
