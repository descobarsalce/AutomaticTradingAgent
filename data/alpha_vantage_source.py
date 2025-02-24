import logging
import pandas as pd
from datetime import datetime
from data.base_sources import DataSource
from data.stock_downloader import StockDownloader

logger = logging.getLogger(__name__)

class AlphaVantageSource(DataSource):
    """Alpha Vantage implementation using StockDownloader."""
    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            downloader = StockDownloader(
                start_date=start_date.date(),
                end_date=end_date.date(),
                source='alpha_vantage'
            )
            return downloader.download_stock_data(symbol)
        except Exception as e:
            return pd.DataFrame()