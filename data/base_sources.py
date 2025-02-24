
"""Base classes for data sources."""
from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd

class DataSource(ABC):
    """Abstract base class for data sources."""
    @abstractmethod
    def fetch_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        pass
