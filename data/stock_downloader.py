
import yfinance as yf
import pandas as pd
from datetime import datetime, date
from alpha_vantage.timeseries import TimeSeries
import os

class StockDownloader:
    def __init__(self, start_date: date, end_date: date, source: str = 'yahoo'):
        self.source = source
        # Convert dates to proper format based on source
        if source == 'alpha_vantage':
            # Alpha Vantage needs datetime
            self.start_date = datetime.combine(start_date, datetime.min.time())
            self.end_date = datetime.combine(end_date, datetime.max.time())
            self.av_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            if not self.av_api_key:
                raise ValueError("Alpha Vantage API key not found in environment variables")
        else:
            # Yahoo Finance works with dates
            self.start_date = start_date
            self.end_date = end_date

    def download_stock_data(self, symbol: str) -> pd.DataFrame:
        """
        Downloads stock data for a given symbol using selected source.

        Args:
            symbol (str): Stock symbol to download

        Returns:
            pd.DataFrame: DataFrame containing the stock data
        """
        try:
            if self.source == 'yahoo':
                return self._download_yahoo(symbol)
            elif self.source == 'alpha_vantage':
                return self._download_alpha_vantage(symbol)
            else:
                raise ValueError(f"Unsupported data source: {self.source}")

        except Exception as e:
            raise Exception(f"Failed to download data for {symbol}: {str(e)}")

    def _download_yahoo(self, symbol: str) -> pd.DataFrame:
        """Download data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=self.start_date,
                end=self.end_date,
                interval="1d"
            )

            if df.empty:
                return None

            return self._process_data(df)
        except Exception as e:
            raise Exception(f"Yahoo Finance error: {str(e)}")

    def _download_alpha_vantage(self, symbol: str) -> pd.DataFrame:
        """Download data from Alpha Vantage"""
        try:
            ts = TimeSeries(key=self.av_api_key, output_format='pandas')
            data, _ = ts.get_daily(symbol=symbol, outputsize='full')

            # Filter data based on date range
            data.index = pd.to_datetime(data.index)
            mask = (data.index >= self.start_date) & (data.index <= self.end_date)
            df = data[mask]

            if df.empty:
                return None

            # Rename columns to match Yahoo Finance format
            df = df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })

            return self._process_data(df)
        except Exception as e:
            raise Exception(f"Alpha Vantage error: {str(e)}")

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and processes the downloaded data.

        Args:
            df (pd.DataFrame): Raw DataFrame

        Returns:
            pd.DataFrame: Processed DataFrame
        """
        # Remove any duplicate indices
        df = df[~df.index.duplicated(keep='first')]

        # Forward fill missing values
        df = df.ffill()

        # Calculate additional metrics
        df['Daily_Return'] = df['Close'].pct_change()
        df['Trading_Volume'] = df['Volume'].fillna(0).astype(int)

        # Round numeric columns to 4 decimal places
        numeric_columns = df.select_dtypes(include=['float64']).columns
        df[numeric_columns] = df[numeric_columns].round(4)

        return df
