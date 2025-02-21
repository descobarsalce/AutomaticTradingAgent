"""Script to load historical stock data for MSFT with validation."""

import yfinance as yf
from datetime import datetime, timedelta
from data.data_handler import DataHandler
import pandas as pd
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_stock_data(data: pd.DataFrame, symbol: str) -> bool:
    """Validate downloaded stock data."""
    if data.empty:
        return False

    required_cols = [f'{col}_{symbol}' for col in ['Open', 'High', 'Low', 'Close', 'Volume']]
    if not all(col in data.columns for col in required_cols):
        return False

    # Check for missing values and suspicious values
    if data[required_cols].isnull().sum().sum() > 0:
        return False

    return True

def main():
    data_handler = DataHandler()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)

    symbol = 'AMD'
    logger.info(f"Starting data collection for {symbol}")

    try:
        data = data_handler.fetch_data([symbol], start_date, end_date)

        if validate_stock_data(data, symbol):
            logger.info(f"Successfully downloaded and validated data for {symbol}")
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
        else:
            logger.error(f"Invalid data received for {symbol}")

    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")

if __name__ == "__main__":
    main()