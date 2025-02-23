ticker = yf.Ticker(symbol)
df = ticker.history(period="1mo", interval="1h")
df

