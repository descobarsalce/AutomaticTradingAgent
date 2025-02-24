
import logging
import pandas as pd
from datetime import datetime
import yfinance as yf
from data.base_sources import DataSource

logger = logging.getLogger(__name__)

class YFinanceSource(DataSource):
    """YFinance implementation of data source with improved reliability."""
    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch stock data using yfinance with better error handling
        """
        try:
            logger.info(f"Fetching daily data for {symbol} from {start_date} to {end_date}")
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval='1d'
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Process the data
            df = df.copy()
            df.index.name = 'Date'
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df = df.round(2)
            
            logger.info(f"Successfully downloaded {len(df)} records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
