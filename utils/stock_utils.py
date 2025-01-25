"""
Utility functions for stock data processing and manipulation
"""
from typing import List

def parse_stock_list(stock_string: str) -> List[str]:
    """
    Parse a comma-separated string of stock symbols into a list
    
    Args:
        stock_string: Comma-separated string of stock symbols
        
    Returns:
        List of cleaned, uppercase stock symbols
    """
    # Handle empty or None input
    if not stock_string or not isinstance(stock_string, str):
        return []

    # Split by comma and clean each stock symbol
    stocks = [
        stock.strip().upper()  # Convert to uppercase and remove whitespace
        for stock in stock_string.split(',')
        if stock.strip()  # Skip empty entries
    ]

    # Remove duplicates while preserving order
    seen = set()
    unique_stocks = [
        stock for stock in stocks if not (stock in seen or seen.add(stock))
    ]

    return unique_stocks
