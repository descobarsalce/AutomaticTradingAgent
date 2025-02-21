
"""Script to load historical stock data for US stocks in batches."""

import yfinance as yf
from datetime import datetime, timedelta
from data.data_handler import DataHandler
import pandas as pd
import time
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sp500_symbols():
    """Get S&P 500 stock symbols."""
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        return sp500['Symbol'].tolist()[:100]  # Get first 100 symbols
    except Exception as e:
        logger.error(f"Error fetching S&P 500 symbols: {e}")
        return []

def main():
    # Initialize DataHandler
    data_handler = DataHandler()
    
    # Set date range for last 10 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)
    
    # Get stock symbols
    symbols = get_sp500_symbols()
    if not symbols:
        logger.error("Failed to get stock symbols. Exiting.")
        return
    
    logger.info(f"Starting data collection for {len(symbols)} stocks")
    
    # Process stocks with progress bar
    failed_symbols = []
    for symbol in tqdm(symbols, desc="Processing stocks"):
        try:
            # Add delay between requests to avoid rate limiting
            time.sleep(1)
            
            logger.info(f"Fetching data for {symbol}")
            data = data_handler.fetch_data([symbol], start_date, end_date)
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                failed_symbols.append(symbol)
                continue
                
            logger.info(f"Successfully processed {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            failed_symbols.append(symbol)
            continue
    
    # Report results
    logger.info("Data collection completed")
    logger.info(f"Successfully processed: {len(symbols) - len(failed_symbols)} stocks")
    if failed_symbols:
        logger.warning(f"Failed symbols: {failed_symbols}")

if __name__ == "__main__":
    main()
