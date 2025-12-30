"""Tests for feature sources."""

import pytest
import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
from datetime import datetime, timedelta

from src.data.feature_engineering.sources.market_data_source import MarketDataSource
from src.data.feature_engineering.sources.technical_source import TechnicalSource


def create_sample_data(n_days: int = 100, symbols: list = None) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    symbols = symbols or ["AAPL"]
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="D")

    data = {}
    for symbol in symbols:
        np.random.seed(42)  # For reproducibility
        base_price = 100
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = base_price * np.exp(np.cumsum(returns))

        data[f"Open_{symbol}"] = prices * (1 + np.random.uniform(-0.01, 0.01, n_days))
        data[f"High_{symbol}"] = prices * (1 + np.random.uniform(0, 0.02, n_days))
        data[f"Low_{symbol}"] = prices * (1 - np.random.uniform(0, 0.02, n_days))
        data[f"Close_{symbol}"] = prices
        data[f"Volume_{symbol}"] = np.random.randint(1000000, 10000000, n_days)

    return pd.DataFrame(data, index=dates)


class TestMarketDataSource:
    """Test suite for MarketDataSource."""

    def test_get_available_features(self):
        """Test listing available features."""
        source = MarketDataSource()
        features = source.get_available_features()

        assert len(features) > 0
        assert "returns" in features
        assert "price_change" in features
        assert "volume_change" in features

    def test_compute_basic_features(self):
        """Test computing basic price features."""
        source = MarketDataSource()
        data = create_sample_data(50)

        result = source.compute_features(data, ["AAPL"], ["returns", "price_change"])

        assert "returns_AAPL" in result.columns
        assert "price_change_AAPL" in result.columns
        assert len(result) == 50

    def test_compute_rolling_features(self):
        """Test computing rolling statistics."""
        source = MarketDataSource(config={"windows": [5, 10]})
        data = create_sample_data(50)

        features = ["rolling_mean_5", "rolling_std_10"]
        result = source.compute_features(data, ["AAPL"], features)

        assert "rolling_mean_5_AAPL" in result.columns
        assert "rolling_std_10_AAPL" in result.columns

    def test_compute_lagged_features(self):
        """Test computing lagged features."""
        source = MarketDataSource(config={"lags": 3})
        data = create_sample_data(50)

        features = ["close_lag1", "close_lag2", "close_lag3"]
        result = source.compute_features(data, ["AAPL"], features)

        for lag in [1, 2, 3]:
            assert f"close_lag{lag}_AAPL" in result.columns

    def test_multiple_symbols(self):
        """Test computing features for multiple symbols."""
        source = MarketDataSource()
        data = create_sample_data(50, ["AAPL", "MSFT"])

        result = source.compute_features(data, ["AAPL", "MSFT"], ["returns"])

        assert "returns_AAPL" in result.columns
        assert "returns_MSFT" in result.columns

    def test_validate_data(self):
        """Test data validation."""
        source = MarketDataSource()

        # Valid data
        valid_data = create_sample_data(50)
        assert source.validate_data(valid_data, ["AAPL"])

        # Invalid data - empty
        empty_data = pd.DataFrame()
        assert not source.validate_data(empty_data, ["AAPL"])


class TestTechnicalSource:
    """Test suite for TechnicalSource."""

    def test_get_available_features(self):
        """Test listing available technical features."""
        source = TechnicalSource()
        features = source.get_available_features()

        assert "rsi" in features
        assert "macd" in features
        assert "bb_upper" in features
        assert "volatility" in features

    def test_compute_rsi(self):
        """Test RSI computation."""
        source = TechnicalSource(config={"rsi_period": 14})
        data = create_sample_data(50)

        result = source.compute_features(data, ["AAPL"], ["rsi"])

        assert "rsi_AAPL" in result.columns
        # RSI should be between 0 and 100
        valid_rsi = result["rsi_AAPL"].dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

    def test_compute_macd(self):
        """Test MACD computation."""
        source = TechnicalSource()
        data = create_sample_data(50)

        result = source.compute_features(
            data, ["AAPL"],
            ["macd", "macd_signal", "macd_histogram"]
        )

        assert "macd_AAPL" in result.columns
        assert "macd_signal_AAPL" in result.columns
        assert "macd_histogram_AAPL" in result.columns

    def test_compute_bollinger_bands(self):
        """Test Bollinger Bands computation."""
        source = TechnicalSource()
        data = create_sample_data(50)

        result = source.compute_features(
            data, ["AAPL"],
            ["bb_upper", "bb_lower", "bb_sma"]
        )

        assert "bb_upper_AAPL" in result.columns
        assert "bb_lower_AAPL" in result.columns
        assert "bb_sma_AAPL" in result.columns

        # Upper should be above lower
        valid_idx = result["bb_upper_AAPL"].notna() & result["bb_lower_AAPL"].notna()
        assert (result.loc[valid_idx, "bb_upper_AAPL"] >= result.loc[valid_idx, "bb_lower_AAPL"]).all()

    def test_compute_stochastic(self):
        """Test Stochastic Oscillator computation."""
        source = TechnicalSource()
        data = create_sample_data(50)

        result = source.compute_features(data, ["AAPL"], ["stoch_k", "stoch_d"])

        assert "stoch_k_AAPL" in result.columns
        assert "stoch_d_AAPL" in result.columns

    def test_compute_volatility(self):
        """Test volatility computation."""
        source = TechnicalSource()
        data = create_sample_data(50)

        result = source.compute_features(data, ["AAPL"], ["volatility"])

        assert "volatility_AAPL" in result.columns
        # Volatility should be non-negative
        valid_vol = result["volatility_AAPL"].dropna()
        assert (valid_vol >= 0).all()

    def test_dependencies(self):
        """Test source dependencies."""
        source = TechnicalSource()

        assert "Close" in source.dependencies

    def test_metadata(self):
        """Test source metadata."""
        source = TechnicalSource()
        metadata = source.metadata

        assert "name" in metadata
        assert "version" in metadata
        assert "feature_count" in metadata
        assert metadata["feature_count"] > 0
