
"""Simple script to test downloading stock data from yfinance."""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Optional, List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_stock_data(
    symbol: str,
    period: str = "1mo",
    interval: str = "1h"
) -> Optional[pd.DataFrame]:
    """
    Fetch stock data using yfinance with error handling
    
    Args:
        symbol: Stock ticker symbol
        period: Data period (e.g. '1d', '1mo', '1y')
        interval: Data interval (e.g. '1m', '1h', '1d')
        
    Returns:
        DataFrame with stock data or None if error
    """
    try:
        logger.info(f"Fetching {interval} data for {symbol} over {period}")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return None

        # Process the data
        df = df.copy()
        df.index.name = 'Date'
        df = df.round(2)
        
        logger.info(f"Successfully downloaded {len(df)} records for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def bulk_download(
    symbols: List[str],
    period: str = "1mo",
    interval: str = "1h"
) -> Dict[str, pd.DataFrame]:
    """
    Download data for multiple symbols
    
    Args:
        symbols: List of stock symbols
        period: Data period
        interval: Data interval
        
    Returns:
        Dictionary of symbol -> DataFrame
    """
    results = {}
    for symbol in symbols:
        df = get_stock_data(symbol, period, interval)
        if df is not None:
            results[symbol] = df
    return results

def main():
    # Test symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    period = "1mo"
    interval = "1h"
    
    logger.info("Starting data download test...")
    
    # Download data
    data = bulk_download(symbols, period, interval)
    
    # Print summary
    for symbol, df in data.items():
        print(f"\nSummary for {symbol}:")
        print(f"Data points: {len(df)}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"Columns: {', '.join(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())

if __name__ == "__main__":
    main()
