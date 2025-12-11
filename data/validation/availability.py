"""Helpers for annotating OHLCV frames with availability metadata."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd


def annotate_availability(aligned: pd.DataFrame,
                          validated_index: pd.DatetimeIndex,
                          symbols: List[str]) -> pd.DataFrame:
    """Attach availability labels and release timestamps for each column."""
    availability: Dict[str, str] = {}
    release_times: Dict[str, pd.Series] = {}

    validated_index_series = validated_index.to_series()
    aligned_index = aligned.index

    open_release = validated_index_series.iloc[1:]
    shifted_release = validated_index_series.shift(1).iloc[1:]
    open_release.index = aligned_index
    shifted_release.index = aligned_index

    for symbol in symbols:
        open_col = f"Open_{symbol}"
        availability[open_col] = "open"
        release_times[open_col] = open_release.copy()

        for col in ["High", "Low", "Close", "Volume"]:
            col_name = f"{col}_{symbol}"
            availability[col_name] = "close"
            release_times[col_name] = shifted_release.copy()

    aligned.attrs["availability"] = availability
    aligned.attrs["release_times"] = release_times
    return aligned
