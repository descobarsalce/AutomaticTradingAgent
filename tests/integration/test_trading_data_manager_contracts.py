from datetime import datetime

import pytest

pd = pytest.importorskip("pandas")

from src.data.data_handler import TradingDataManager


class DummyProvider:
    def __init__(self, frame):
        self._frame = frame
        self.last_args = None

    def fetch(self, symbols, start, end):
        # store arguments to assert timezone handling
        self.last_args = (symbols, start, end)
        return self._frame.copy()


def _build_frame(symbol="AAPL"):
    index = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
    data = {
        f"Open_{symbol}": [101, 99, 100],
        f"High_{symbol}": [102, 100, 101],
        f"Low_{symbol}": [98, 97, 99],
        f"Close_{symbol}": [100.5, 98.5, 99.5],
        f"Volume_{symbol}": [1500, 1200, 1300],
    }
    return pd.DataFrame(data, index=index)


def test_manager_enforces_validation_and_availability_annotations():
    frame = _build_frame()
    provider = DummyProvider(frame)
    manager = TradingDataManager(provider)

    result = manager.fetch(["AAPL"], datetime(2024, 1, 1), datetime(2024, 1, 3))

    symbols_arg, start_arg, end_arg = provider.last_args
    assert symbols_arg == ["AAPL"]
    assert str(start_arg.tz) == "UTC"
    assert str(end_arg.tz) == "UTC"

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.is_monotonic_increasing
    assert str(result.index.tz) == "UTC"

    availability = result.attrs.get("availability")
    release_times = result.attrs.get("release_times")

    assert availability["Open_AAPL"] == "open"
    assert availability["Close_AAPL"] == "close"
    pd.testing.assert_index_equal(result.index, release_times["Open_AAPL"].index)
    pd.testing.assert_index_equal(result.index, release_times["Close_AAPL"].index)


def test_manager_propagates_validation_errors_for_missing_columns():
    frame = _build_frame().drop(columns=["Volume_AAPL"])
    provider = DummyProvider(frame)
    manager = TradingDataManager(provider)

    with pytest.raises(ValueError):
        manager.fetch(["AAPL"], datetime(2024, 1, 1), datetime(2024, 1, 3))
