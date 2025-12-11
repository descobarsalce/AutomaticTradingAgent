from datetime import datetime, timezone

import pytest

pd = pytest.importorskip("pandas")

from data.validation import (
    annotate_availability,
    ensure_datetime_index,
    ensure_utc_timestamp,
    validate_ohlcv_frame,
)


@pytest.fixture
def sample_frame():
    index = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
    data = {
        "Open_AAPL": [101, 99, 100],
        "High_AAPL": [102, 100, 101],
        "Low_AAPL": [98, 97, 99],
        "Close_AAPL": [100.5, 98.5, 99.5],
        "Volume_AAPL": [1500, 1200, 1300],
    }
    frame = pd.DataFrame(data, index=index)
    return frame


def test_ensure_utc_timestamp_handles_naive_and_tzaware():
    naive = datetime(2024, 1, 1)
    aware = datetime(2024, 1, 1, tzinfo=timezone.utc)

    naive_ts = ensure_utc_timestamp(naive)
    aware_ts = ensure_utc_timestamp(aware)

    assert naive_ts.tzinfo is not None
    assert aware_ts.tzinfo is not None
    assert str(naive_ts.tz) == "UTC"
    assert str(aware_ts.tz) == "UTC"


def test_ensure_datetime_index_sorts_and_localizes(sample_frame):
    ensured = ensure_datetime_index(sample_frame)

    assert isinstance(ensured.index, pd.DatetimeIndex)
    assert ensured.index.is_monotonic_increasing
    assert str(ensured.index.tz) == "UTC"


def test_validate_ohlcv_frame_enforces_required_columns(sample_frame):
    ordered = ensure_datetime_index(sample_frame)
    validated = validate_ohlcv_frame(ordered, ["AAPL"])

    assert validated.equals(ordered)

    missing = ordered.drop(columns=["Close_AAPL"])
    with pytest.raises(ValueError):
        validate_ohlcv_frame(missing, ["AAPL"])


def test_annotate_availability_attaches_metadata(sample_frame):
    ordered = ensure_datetime_index(sample_frame)
    validated = validate_ohlcv_frame(ordered, ["AAPL"])
    aligned = validated.copy()
    aligned.index = aligned.index.shift(-1, freq="D")[:-1]

    annotated = annotate_availability(aligned, validated.index, ["AAPL"])

    availability = annotated.attrs.get("availability")
    release_times = annotated.attrs.get("release_times")

    assert availability is not None
    assert release_times is not None
    assert availability["Open_AAPL"] == "open"
    assert availability["Close_AAPL"] == "close"
    pd.testing.assert_index_equal(
        annotated.index,
        release_times["Open_AAPL"].index,
    )
    pd.testing.assert_index_equal(
        annotated.index,
        release_times["Close_AAPL"].index,
    )
