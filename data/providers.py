"""Data provider interface and concrete implementations."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Protocol, runtime_checkable

import pandas as pd

from data.data_handler import (
    DataHandler,
    ensure_utc_timestamp,
    validate_ohlcv_frame,
)


@runtime_checkable
class DataProvider(Protocol):
    """Protocol for fetching market data with OHLCV schema enforcement."""

    def fetch(self, symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:  # pragma: no cover - interface
        ...


class SessionStateProvider:
    """Provider that delegates to the Streamlit session state's data handler."""

    def __init__(self, data_handler: DataHandler | None = None):
        # Lazy import to avoid Streamlit dependency during testing when unused
        import streamlit as st

        self._data_handler = data_handler or st.session_state.data_handler

    def fetch(self, symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:
        frame = self._data_handler.fetch_data(symbols, start, end)
        return validate_ohlcv_frame(frame, symbols)


class LocalCacheProvider:
    """Provider that reads from the local SQL cache before falling back to downloads."""

    def __init__(self, data_handler: DataHandler | None = None):
        self._data_handler = data_handler or DataHandler()

    def fetch(self, symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:
        frame = self._data_handler.fetch_data(symbols, start, end, use_SQL=True)
        return validate_ohlcv_frame(frame, symbols)


class LiveAPIProvider:
    """Provider that always fetches from live APIs, bypassing cached results."""

    def __init__(self, data_handler: DataHandler | None = None):
        self._data_handler = data_handler or DataHandler()

    def fetch(self, symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:
        frame = self._data_handler.fetch_data(symbols, start, end, use_SQL=False)
        return validate_ohlcv_frame(frame, symbols)


class FileSystemProvider:
    """Provider that serves OHLCV data from cached parquet or CSV files."""

    def __init__(self, file_path: str | Path):
        self._path = Path(file_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Data file not found: {self._path}")

    def _load_frame(self) -> pd.DataFrame:
        if self._path.suffix.lower() == ".parquet":
            return pd.read_parquet(self._path)
        if self._path.suffix.lower() == ".csv":
            return pd.read_csv(self._path, parse_dates=["Date"], infer_datetime_format=True)
        raise ValueError("FileSystemProvider supports only parquet and CSV files")

    def fetch(self, symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:
        frame = self._load_frame()
        if "Date" in frame.columns and frame.index.name != "Date":
            frame = frame.set_index("Date")

        frame = validate_ohlcv_frame(frame, symbols)

        start_ts = ensure_utc_timestamp(start)
        end_ts = ensure_utc_timestamp(end)

        return frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)]
