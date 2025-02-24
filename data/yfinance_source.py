from datetime import datetime, date
import pandas as pd
from data.base_sources import DataSource
from data.stock_downloader import StockDownloader

class YFinanceSource(DataSource):
    """YFinance implementation using StockDownloader."""
    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            downloader = StockDownloader(
                start_date=start_date.date(),
                end_date=end_date.date(),
                source='yahoo'
            )
            return downloader.download_stock_data(symbol)
        except Exception as e:
            return pd.DataFrame()