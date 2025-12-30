"""Tests for the FeatureEngineer orchestrator."""

import pytest
import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
from datetime import datetime

from src.data.feature_engineering import FeatureEngineer
from src.data.feature_engineering.sources.market_data_source import MarketDataSource
from src.data.feature_engineering.sources.technical_source import TechnicalSource


def create_sample_data(n_days: int = 100, symbols: list = None) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    symbols = symbols or ["AAPL"]
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="D")

    data = {}
    for symbol in symbols:
        np.random.seed(42)
        base_price = 100
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = base_price * np.exp(np.cumsum(returns))

        data[f"Open_{symbol}"] = prices * (1 + np.random.uniform(-0.01, 0.01, n_days))
        data[f"High_{symbol}"] = prices * (1 + np.random.uniform(0, 0.02, n_days))
        data[f"Low_{symbol}"] = prices * (1 - np.random.uniform(0, 0.02, n_days))
        data[f"Close_{symbol}"] = prices
        data[f"Volume_{symbol}"] = np.random.randint(1000000, 10000000, n_days)

    return pd.DataFrame(data, index=dates)


class TestFeatureEngineer:
    """Test suite for FeatureEngineer."""

    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer(enable_cache=False)

        assert engineer.registry is not None
        assert engineer.engine is not None

    def test_register_source(self):
        """Test registering a source."""
        engineer = FeatureEngineer(enable_cache=False)
        source = MarketDataSource()

        engineer.register_source(source)

        assert "MarketDataSource" in engineer.list_sources()

    def test_register_default_sources(self):
        """Test registering default sources."""
        engineer = FeatureEngineer(enable_cache=False)
        engineer.register_default_sources()

        sources = engineer.list_sources()
        assert "MarketDataSource" in sources
        assert "TechnicalSource" in sources

    def test_compute_features_basic(self):
        """Test basic feature computation."""
        engineer = FeatureEngineer(enable_cache=False)
        engineer.register_default_sources()

        data = create_sample_data(100)
        result = engineer.compute_features(data, symbols=["AAPL"])

        # Should have original columns plus computed features
        assert len(result.columns) > len(data.columns)
        # Should have some technical indicators
        assert any("rsi" in col.lower() for col in result.columns)

    def test_compute_features_multiple_symbols(self):
        """Test computing features for multiple symbols."""
        engineer = FeatureEngineer(enable_cache=False)
        engineer.register_default_sources()

        data = create_sample_data(100, ["AAPL", "MSFT"])
        result = engineer.compute_features(data, symbols=["AAPL", "MSFT"])

        # Should have features for both symbols
        assert any("AAPL" in col for col in result.columns)
        assert any("MSFT" in col for col in result.columns)

    def test_auto_detect_symbols(self):
        """Test automatic symbol detection."""
        engineer = FeatureEngineer(enable_cache=False)
        engineer.register_default_sources()

        data = create_sample_data(100, ["AAPL", "GOOGL"])
        # Don't specify symbols - should auto-detect
        result = engineer.compute_features(data)

        assert any("AAPL" in col for col in result.columns)
        assert any("GOOGL" in col for col in result.columns)

    def test_compute_specific_sources(self):
        """Test computing from specific sources only."""
        engineer = FeatureEngineer(enable_cache=False)
        engineer.register_default_sources()

        data = create_sample_data(100)
        result = engineer.compute_features(
            data,
            symbols=["AAPL"],
            sources=["MarketDataSource"],
        )

        # Should have market data features but not technical ones
        assert any("returns" in col.lower() for col in result.columns)

    def test_list_features(self):
        """Test listing available features."""
        engineer = FeatureEngineer(enable_cache=False)
        engineer.register_default_sources()

        features = engineer.list_features()
        assert len(features) > 0

    def test_list_features_by_source(self):
        """Test listing features by source."""
        engineer = FeatureEngineer(enable_cache=False)
        engineer.register_default_sources()

        market_features = engineer.list_features("MarketDataSource")
        technical_features = engineer.list_features("TechnicalSource")

        assert len(market_features) > 0
        assert len(technical_features) > 0
        assert "returns" in market_features
        assert "rsi" in technical_features

    def test_current_index_slicing(self):
        """Test data slicing with current_index."""
        engineer = FeatureEngineer(enable_cache=False)
        engineer.register_default_sources()

        data = create_sample_data(100)
        result = engineer.compute_features(data, current_index=50)

        # Result should be shorter (after dropna)
        assert len(result) <= 51

    def test_drop_na(self):
        """Test dropping NaN values."""
        engineer = FeatureEngineer(enable_cache=False)
        engineer.register_default_sources()

        data = create_sample_data(100)
        result = engineer.compute_features(data, drop_na=True)

        # No NaN values should remain
        assert not result.isna().any().any()

    def test_get_stats(self):
        """Test getting statistics."""
        engineer = FeatureEngineer(enable_cache=False)
        engineer.register_default_sources()

        stats = engineer.get_stats()
        assert "registry" in stats
        assert "engine" in stats

    def test_legacy_api_prepare_data(self):
        """Test backward compatible prepare_data method."""
        engineer = FeatureEngineer(enable_cache=False)
        engineer.register_default_sources()

        data = create_sample_data(100)
        result = engineer.prepare_data(data)

        assert len(result.columns) > len(data.columns)

    def test_from_legacy_factory(self):
        """Test from_legacy factory method."""
        engineer = FeatureEngineer.from_legacy()

        assert "MarketDataSource" in engineer.list_sources()
        assert "TechnicalSource" in engineer.list_sources()

    def test_invalid_data_raises_error(self):
        """Test that invalid data raises ValueError."""
        engineer = FeatureEngineer(enable_cache=False)
        engineer.register_default_sources()

        with pytest.raises(ValueError):
            engineer.compute_features(None)

        with pytest.raises(ValueError):
            engineer.compute_features(pd.DataFrame())

    def test_repr(self):
        """Test string representation."""
        engineer = FeatureEngineer(enable_cache=False)
        engineer.register_default_sources()

        repr_str = repr(engineer)
        assert "FeatureEngineer" in repr_str
        assert "sources" in repr_str
