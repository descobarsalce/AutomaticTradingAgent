def fetch_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch data with improved caching, validation and rate limiting."""
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(',') if s.strip()]
    elif isinstance(symbols, list):
        symbols = [s.strip() for s in symbols if s.strip()]

    if not symbols:
        raise ValueError("No symbols provided")

    # Validate symbols
    if not all(isinstance(s, str) and s for s in symbols):
        raise ValueError("Invalid symbol format")

    all_stocks_data = pd.DataFrame()
    failed_symbols = []

    for symbol in symbols:
        try:
            # Check cache first
            cached_data = self._sql_handler.get_cached_data(symbol, start_date, end_date)

            if cached_data is not None and not cached_data.empty:
                stock_data = cached_data
                logger.info(f"Using cached data for {symbol}")
            else:
                logger.info(f"Fetching new data for {symbol}")
                stock_data = self._data_source.fetch_data(symbol, start_date, end_date)
                    if not stock_data.empty:
                        break

                if not stock_data.empty:
                    self._sql_handler.cache_data(symbol, stock_data, start_date, end_date)
                else:
                    logger.warning(f"Empty data received for {symbol} after retries")
                    failed_symbols.append(symbol)
                    continue

            # Validate data
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in stock_data.columns for col in required_cols):
                logger.error(f"Missing required columns for {symbol}")
                failed_symbols.append(symbol)
                continue

            # Suffix columns with symbol
            stock_data.columns = [f'{col}_{symbol}' for col in stock_data.columns]
            all_stocks_data = pd.concat([all_stocks_data, stock_data], axis=1)
            logger.info(f"Successfully processed data for {symbol}")

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            failed_symbols.append(symbol)
            continue

    if all_stocks_data.empty:
        raise ValueError(f"No valid data retrieved for any symbols. Failed symbols: {failed_symbols}")
    elif failed_symbols:
        logger.warning(f"Data retrieval failed for symbols: {failed_symbols}")

    return all_stocks_data
