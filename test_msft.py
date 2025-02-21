
from data.data_handler import DataHandler
from datetime import datetime, timedelta
import pandas as pd

# Initialize data handler
handler = DataHandler()

# Set date range for last 30 days
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

# Fetch MSFT data
try:
    df = handler.fetch_data(['MSFT'], start_date, end_date)
    print("\nMSFT Data Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    print("\nDate Range:", df.index.min(), "to", df.index.max())
except Exception as e:
    print(f"Error: {str(e)}")
