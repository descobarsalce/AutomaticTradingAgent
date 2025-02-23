
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tesla_fetch():
    try:
        symbol = "TSLA"
        ticker = yf.Ticker(symbol)
        
        # Try up to 3 times with increasing periods
        periods = ["2mo", "3mo", "6mo"]
        
        for period in periods:
            time.sleep(1)  # Respect rate limits
            df = ticker.history(period=period, interval="1h")
            
            if not df.empty:
                logger.info(f"Successfully fetched {len(df)} rows of {symbol} data using period={period}")
                logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
                return df
            
            logger.warning(f"Empty dataset for period {period}, trying next period...")
        
        logger.error(f"Failed to fetch data for {symbol} after trying all periods")
        return None
        
    except Exception as e:
        logger.error(f"Error fetching {symbol} data: {str(e)}")
        return None

if __name__ == "__main__":
    result = test_tesla_fetch()
    if result is not None:
        print("\nLatest 5 records:")
        print(result.tail())
