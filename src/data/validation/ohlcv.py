"""OHLCV dataframe validation helpers."""
from __future__ import annotations

from typing import List

import pandas as pd

from src.data.validation.time_utils import ensure_datetime_index

REQUIRED_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def validate_ohlcv_frame(frame: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """Validate OHLCV schema, non-null columns, and timezone-aware index."""
    if frame is None or frame.empty:
        raise ValueError("Empty data provided")

    frame = ensure_datetime_index(frame)

    for symbol in symbols:
        for col in REQUIRED_OHLCV_COLUMNS:
            col_name = f"{col}_{symbol}"
            if col_name not in frame.columns:
                raise ValueError(f"Missing column {col_name}")
            if frame[col_name].isna().any():
                raise ValueError(f"Column {col_name} contains null values")

        open_col = f"Open_{symbol}"
        high_col = f"High_{symbol}"
        low_col = f"Low_{symbol}"
        close_col = f"Close_{symbol}"
        volume_col = f"Volume_{symbol}"

        if (frame[[open_col, high_col, low_col, close_col]] <= 0).any().any():
            raise ValueError(f"Non-positive price detected for {symbol}")
        if (frame[volume_col] < 0).any():
            raise ValueError(f"Negative volume detected for {symbol}")

        if (frame[high_col] < frame[[open_col, close_col]].max(axis=1)).any():
            raise ValueError(f"High price below open/close for {symbol}")
        if (frame[low_col] > frame[[open_col, close_col]].min(axis=1)).any():
            raise ValueError(f"Low price above open/close for {symbol}")

    return frame
