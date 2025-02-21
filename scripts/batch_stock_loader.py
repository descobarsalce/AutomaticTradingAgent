
"""Script to load historical stock data for US stocks in batches with validation."""

import yfinance as yf
from datetime import datetime, timedelta
from data.data_handler import DataHandler
import pandas as pd
import time
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

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

def validate_stock_data(data: pd.DataFrame, symbol: str) -> bool:
    """Validate downloaded stock data."""
    if data.empty:
        return False
        
    required_cols = [f'{col}_{symbol}' for col in ['Open', 'High', 'Low', 'Close', 'Volume']]
    if not all(col in data.columns for col in required_cols):
        return False
        
    # Check for missing values
    if data[required_cols].isnull().sum().sum() > 0:
        return False
        
    # Check for suspicious values
    price_cols = [f'{col}_{symbol}' for col in ['Open', 'High', 'Low', 'Close']]
    if (data[price_cols] <= 0).any().any():
        return False
        
    return True

def process_symbol(symbol: str, data_handler: DataHandler, start_date: datetime, end_date: datetime) -> tuple:
    """Process a single symbol with retries."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(1 + attempt)  # Exponential backoff
            data = data_handler.fetch_data([symbol], start_date, end_date)
            
            if validate_stock_data(data, symbol):
                return (symbol, data, True)
            
            logger.warning(f"Invalid data for {symbol} (attempt {attempt + 1}/{max_retries})")
            
        except Exception as e:
            logger.error(f"Error processing {symbol} (attempt {attempt + 1}/{max_retries}): {e}")
            
    return (symbol, None, False)

def main():
    data_handler = DataHandler()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)
    
    symbols = get_sp500_symbols()
    if not symbols:
        logger.error("Failed to get stock symbols. Exiting.")
        return
    
    logger.info(f"Starting data collection for {len(symbols)} stocks")
    
    successful_downloads = 0
    failed_symbols = []
    
    # Process symbols in parallel with progress tracking
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_symbol, symbol, data_handler, start_date, end_date) 
                  for symbol in symbols]
        
        for future in tqdm(as_completed(futures), total=len(symbols), desc="Processing stocks"):
            symbol, data, success = future.result()
            if success:
                successful_downloads += 1
            else:
                failed_symbols.append(symbol)
    
    # Report results
    success_rate = (successful_downloads / len(symbols)) * 100
    logger.info(f"Data collection completed:")
    logger.info(f"Success rate: {success_rate:.2f}%")
    logger.info(f"Successfully processed: {successful_downloads} stocks")
    logger.info(f"Failed: {len(failed_symbols)} stocks")
    if failed_symbols:
        logger.warning(f"Failed symbols: {failed_symbols}")

if __name__ == "__main__":
    main()
