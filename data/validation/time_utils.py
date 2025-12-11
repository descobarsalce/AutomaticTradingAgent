"""Datetime handling utilities for validation."""
from datetime import datetime

import pandas as pd


def ensure_utc_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    """Convert a datetime-like object to a timezone-aware UTC Timestamp."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame index is a timezone-aware DatetimeIndex sorted by time."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    frame = frame.sort_index()
    if frame.index.isna().any():
        raise ValueError("DataFrame index contains NaT values")
    return frame
