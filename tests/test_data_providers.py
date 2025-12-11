import pandas as pd
import pytest
from datetime import datetime

from environment.trading_env import TradingEnv, fetch_trading_data
from data.data_handler import TradingDataManager
from data.providers import FileSystemProvider


class DummyProvider:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame
        self.called = False

    def fetch(self, symbols, start, end):
        self.called = True
        return self._frame


def _sample_frame(symbols):
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    data = {}
    for symbol in symbols:
        data[f"Open_{symbol}"] = [100.0, 101.0, 102.0]
        data[f"High_{symbol}"] = [101.0, 102.0, 103.0]
        data[f"Low_{symbol}"] = [99.0, 100.0, 101.0]
        data[f"Close_{symbol}"] = [100.5, 101.5, 102.5]
        data[f"Volume_{symbol}"] = [1000, 1100, 1200]
    return pd.DataFrame(data, index=index)


def test_trading_env_uses_injected_provider():
    symbols = ["AAPL"]
    frame = _sample_frame(symbols)
    provider = DummyProvider(frame)

    env = TradingEnv(
        stock_names=symbols,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 3),
        provider=provider
    )

    assert provider.called is True
    assert env.data.index.tz is not None


def test_trading_env_requires_provider():
    symbols = ["AAPL"]

    with pytest.raises(ValueError):
        TradingEnv(
            stock_names=symbols,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 3),
            provider=None  # type: ignore[arg-type]
        )


def test_validation_missing_column_raises_error():
    symbols = ["MSFT"]
    frame = _sample_frame(symbols)
    frame = frame.drop(columns=["Volume_MSFT"])
    provider = DummyProvider(frame)

    with pytest.raises(ValueError):
        fetch_trading_data(symbols, datetime(2024, 1, 1), datetime(2024, 1, 3), provider)


def test_validation_null_values_raise_error():
    symbols = ["GOOG"]
    frame = _sample_frame(symbols)
    frame.loc[frame.index[0], "Close_GOOG"] = None
    provider = DummyProvider(frame)

    with pytest.raises(ValueError):
        fetch_trading_data(symbols, datetime(2024, 1, 1), datetime(2024, 1, 3), provider)


def test_timezone_is_enforced():
    symbols = ["IBM"]
    frame = _sample_frame(symbols)
    provider = DummyProvider(frame)

    validated = fetch_trading_data(symbols, datetime(2024, 1, 1), datetime(2024, 1, 3), provider)
    assert validated.index.tz is not None
    assert str(validated.index.tz) == "UTC"


def test_filesystem_provider_loads_cached_frame(tmp_path):
    symbols = ["NFLX"]
    frame = _sample_frame(symbols)
    cached = frame.reset_index().rename(columns={"index": "Date"})
    file_path = tmp_path / "ohlcv.csv"
    cached.to_csv(file_path, index=False)

    provider = FileSystemProvider(file_path)
    fetched = provider.fetch(symbols, datetime(2024, 1, 1), datetime(2024, 1, 3))

    assert fetched.index.tz is not None
    pd.testing.assert_frame_equal(fetched, frame.tz_localize("UTC"))


def test_trading_data_manager_applies_alignment_and_metadata():
    symbols = ["META"]
    frame = _sample_frame(symbols)
    provider = DummyProvider(frame)
    manager = TradingDataManager(provider)

    processed = manager.fetch(symbols, datetime(2024, 1, 1), datetime(2024, 1, 3))

    # First row should represent the second calendar day after alignment
    assert len(processed) == len(frame) - 1
    availability = processed.attrs.get("availability", {})
    assert availability[f"Open_{symbols[0]}"] == "open"
    assert availability[f"Close_{symbols[0]}"] == "close"
    release_times = processed.attrs.get("release_times", {})
    assert release_times[f"Open_{symbols[0]}"].iloc[0] == processed.index[0]
    assert release_times[f"Close_{symbols[0]}"].iloc[0] == processed.index[0] - pd.Timedelta(days=1)
    # Close is shifted by one day
    assert processed.iloc[0][f"Close_{symbols[0]}"] == frame.iloc[0][f"Close_{symbols[0]}"]


def test_environment_handles_provider_swap():
    symbols = ["TSLA"]
    frame_one = _sample_frame(symbols)
    frame_two = _sample_frame(symbols) * 2

    provider_one = DummyProvider(frame_one)
    env_one = TradingEnv(
        stock_names=symbols,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 3),
        provider=provider_one
    )

    provider_two = DummyProvider(frame_two)
    env_two = TradingEnv(
        stock_names=symbols,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 3),
        provider=provider_two
    )

    assert provider_one.called and provider_two.called
    assert not env_one.data.equals(env_two.data)
