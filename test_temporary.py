def execute_all_trades(self, stock_names: List[str], price
     
    for symbol in stock_names:
        price = get_current_price(symbol)
        
# First process all sell actions
for symbol, action in zip(stock_names, actions):
if action < 0:  # Only process sells first

qty = self._calculate_trade_quantity(action, symbol, price, max_pct_position_by_asset)
# if not all_zeros:
# logger.info(f"Intended SELL action for {symbol}: action={action}, price={price}, calculated quantity={qty}")
if qty > 0:
if self.execute_trade(symbol, -qty, price, timestamp):  # Note the negative qty for sells
    trades_executed[symbol] = True

    self._update_metrics(price, symbol)