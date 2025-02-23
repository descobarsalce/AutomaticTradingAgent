
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tesla_fetch():
    try:
        symbol = "TSLA"
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1mo", interval="1h")
        
        if df.empty:
            logger.error(f"Empty dataset received for {symbol}")
            return None
            
        logger.info(f"Successfully fetched {len(df)} rows of {symbol} data")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching {symbol} data: {str(e)}")
        return None

if __name__ == "__main__":
    result = test_tesla_fetch()
    if result is not None:
        print(result.head())
