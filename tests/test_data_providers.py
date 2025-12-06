import pandas as pd
import pytest
from datetime import datetime

from environment.trading_env import TradingEnv, fetch_trading_data


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
