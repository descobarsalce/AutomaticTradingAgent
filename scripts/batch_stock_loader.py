
"""Script to load historical stock data with validation."""

from data.data_handler import DataHandler
from datetime import datetime, timedelta
import logging

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
    
    logger.info(f"Starting data download for {len(symbols)} stocks")
    
    try:
        df = handler.fetch_data(symbols, start_date, end_date)
        
        if not df.empty:
            logger.info("\nDownload Summary:")
            logger.info(f"Data Shape: {df.shape}")
            logger.info(f"Date Range: {df.index.min()} to {df.index.max()}")
            logger.info(f"Available Columns: {df.columns.tolist()}")
            logger.info("\nFirst few rows of data:")
            logger.info(df.head())
            
            # Verify we have data for all symbols
            for symbol in symbols:
                if f'Close_{symbol}' in df.columns:
                    logger.info(f"✅ Successfully downloaded data for {symbol}")
                else:
                    logger.warning(f"❌ Failed to download data for {symbol}")
        else:
            logger.error("No data was downloaded")
            
    except Exception as e:
        logger.error(f"Error during data download: {str(e)}")

if __name__ == "__main__":
    main()
