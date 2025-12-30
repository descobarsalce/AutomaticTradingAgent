from dataclasses import src.dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple
import pandas as pd


@dataclass
class TemporalDataSplitter:
    """Create train/validation windows based on available date ranges."""

    validation_fraction: float = 0.2
    min_train_window: int = 30
    min_validation_window: int = 7

    def __post_init__(self) -> None:
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.min_train_window < 1 or self.min_validation_window < 1:
            raise ValueError("Minimum window sizes must be positive")

    def split_dates(
        self,
        start_date: datetime,
        end_date: datetime,
        date_index: Optional[pd.DatetimeIndex] = None,
    ) -> Tuple[Tuple[datetime, datetime], Tuple[datetime, datetime]]:
        if end_date <= start_date:
            raise ValueError("End date must be after start date for splitting")

        total_points = (
            len(date_index) if date_index is not None else (end_date - start_date).days + 1
        )
        if total_points <= self.min_validation_window + self.min_train_window:
            raise ValueError("Not enough data to create train and validation windows")

        split_idx = max(
            self.min_train_window,
            int(total_points * (1 - self.validation_fraction)),
        )
        if date_index is not None:
            train_end = date_index[min(split_idx, len(date_index) - 2)]
            val_start = date_index[min(split_idx + 1, len(date_index) - 1)]
            val_end = date_index[-1]
            train_range = (date_index[0].to_pydatetime(), train_end.to_pydatetime())
            val_range = (val_start.to_pydatetime(), val_end.to_pydatetime())
        else:
            train_end = start_date + timedelta(days=split_idx)
            val_start = train_end + timedelta(days=1)
            train_range = (start_date, min(train_end, end_date - timedelta(days=1)))
            val_range = (val_start, end_date)

        if (val_range[1] - val_range[0]).days < self.min_validation_window:
            raise ValueError("Validation window too small after split")

        return train_range, val_range
