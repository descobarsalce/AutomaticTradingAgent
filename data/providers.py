"""Data provider interface and concrete implementations."""
from __future__ import annotations

from datetime import datetime
from typing import List, Protocol, runtime_checkable

import pandas as pd

from data.data_handler import DataHandler


@runtime_checkable
class DataProvider(Protocol):
    """Protocol for fetching market data."""

    def fetch(self, symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:  # pragma: no cover - interface
        ...


class SessionStateProvider:
    """Provider that delegates to the Streamlit session state's data handler."""

    def __init__(self, data_handler: DataHandler | None = None):
        # Lazy import to avoid Streamlit dependency during testing when unused
        import streamlit as st

        self._data_handler = data_handler or st.session_state.data_handler

    def fetch(self, symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:
        return self._data_handler.fetch_data(symbols, start, end)


class LocalCacheProvider:
    """Provider that reads from the local SQL cache before falling back to downloads."""

    def __init__(self, data_handler: DataHandler | None = None):
        self._data_handler = data_handler or DataHandler()

    def fetch(self, symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:
        return self._data_handler.fetch_data(symbols, start, end, use_SQL=True)


class LiveAPIProvider:
    """Provider that always fetches from live APIs, bypassing cached results."""

    def __init__(self, data_handler: DataHandler | None = None):
        self._data_handler = data_handler or DataHandler()

    def fetch(self, symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:
        return self._data_handler.fetch_data(symbols, start, end, use_SQL=False)
