
from data.data_handler import DataHandler
from datetime import datetime
import pandas as pd

# Initialize data handler
handler = DataHandler()

# Set date range
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 2, 20)

# Fetch TSLA data - passing as a list
try:
    df = handler.fetch_data(['TSLA'], start_date, end_date)
    print("\nData Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
except Exception as e:
    print(f"Error: {str(e)}")
