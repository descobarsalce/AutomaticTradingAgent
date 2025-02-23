
"""Script to load historical stock data with validation."""

from data.data_handler import DataHandler
from datetime import datetime, timedelta
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize data handler
    handler = DataHandler()
    
    # Set date range for last 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)
    
    # Stock symbols to download
    symbols = ['MRVL', 'DPZ', 'ASML', 'CRM']
    
    logger.info(f"Starting data download for: {', '.join(symbols)}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    
    for symbol in symbols:
        try:
            # Download data for single symbol
            df = handler.fetch_data([symbol], start_date, end_date)
            
            if not df.empty and f'Close_{symbol}' in df.columns:
                logger.info(f"✅ {symbol}: Successfully downloaded {len(df)} rows")
                logger.info(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
            else:
                logger.error(f"❌ {symbol}: Failed to download data")
                
        except Exception as e:
            logger.error(f"❌ {symbol}: Error during download - {str(e)}")

if __name__ == "__main__":
    main()
