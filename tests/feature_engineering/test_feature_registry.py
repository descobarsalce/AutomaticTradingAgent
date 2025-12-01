"""Tests for the FeatureRegistry."""

import pytest
from typing import List, Optional
import pandas as pd

from data.feature_engineering.core.feature_registry import FeatureRegistry
from data.feature_engineering.sources.base_source import BaseFeatureSource


class MockSource(BaseFeatureSource):
    """Mock feature source for testing."""

    def __init__(self, name: str = "MockSource", features: List[str] = None, priority: int = 0):
        super().__init__()
        self.name = name
        self._features = features or ["feature_a", "feature_b"]
        self.priority = priority

    def get_available_features(self) -> List[str]:
        return self._features

    def compute_features(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        return pd.DataFrame(index=data.index)


class TestFeatureRegistry:
    """Test suite for FeatureRegistry."""

    def test_register_source(self):
        """Test registering a source."""
        registry = FeatureRegistry()
        source = MockSource()

        registry.register_source(source)

        assert "MockSource" in registry.list_sources()
        assert len(registry) == 2  # feature_a and feature_b

    def test_list_features(self):
        """Test listing all features."""
        registry = FeatureRegistry()
        source = MockSource(features=["feat1", "feat2", "feat3"])

        registry.register_source(source)

        features = registry.list_features()
        assert len(features) == 3
        assert "feat1" in features
        assert "feat2" in features
        assert "feat3" in features

    def test_list_features_by_source(self):
        """Test listing features for a specific source."""
        registry = FeatureRegistry()
        source1 = MockSource(name="Source1", features=["a", "b"])
        source2 = MockSource(name="Source2", features=["c", "d"])

        registry.register_source(source1)
        registry.register_source(source2)

        features_source1 = registry.list_features("Source1")
        features_source2 = registry.list_features("Source2")

        assert features_source1 == ["a", "b"]
        assert features_source2 == ["c", "d"]

    def test_get_source_for_feature(self):
        """Test getting source that provides a feature."""
        registry = FeatureRegistry()
        source = MockSource(features=["my_feature"])

        registry.register_source(source)

        retrieved = registry.get_source("my_feature")
        assert retrieved is not None
        assert retrieved.name == "MockSource"

    def test_priority_conflict_resolution(self):
        """Test that higher priority source wins for same feature."""
        registry = FeatureRegistry()
        low_priority = MockSource(name="LowPriority", features=["shared"], priority=1)
        high_priority = MockSource(name="HighPriority", features=["shared"], priority=10)

        registry.register_source(low_priority)
        registry.register_source(high_priority)

        source = registry.get_source("shared")
        assert source.name == "HighPriority"

    def test_unregister_source(self):
        """Test unregistering a source."""
        registry = FeatureRegistry()
        source = MockSource(features=["temp_feature"])

        registry.register_source(source)
        assert "temp_feature" in registry

        registry.unregister_source("MockSource")
        assert "temp_feature" not in registry
        assert "MockSource" not in registry.list_sources()

    def test_get_enabled_sources(self):
        """Test getting only enabled sources."""
        registry = FeatureRegistry()
        enabled = MockSource(name="Enabled")
        disabled = MockSource(name="Disabled")
        disabled.enabled = False

        registry.register_source(enabled)
        registry.register_source(disabled)

        enabled_sources = registry.get_enabled_sources()
        assert len(enabled_sources) == 1
        assert enabled_sources[0].name == "Enabled"

    def test_get_registry_stats(self):
        """Test registry statistics."""
        registry = FeatureRegistry()
        source1 = MockSource(name="S1", features=["a", "b"])
        source2 = MockSource(name="S2", features=["c"])

        registry.register_source(source1)
        registry.register_source(source2)

        stats = registry.get_registry_stats()
        assert stats["total_sources"] == 2
        assert stats["total_features"] == 3
        assert stats["sources"]["S1"] == 2
        assert stats["sources"]["S2"] == 1

    def test_contains_feature(self):
        """Test feature containment check."""
        registry = FeatureRegistry()
        source = MockSource(features=["exists"])

        registry.register_source(source)

        assert "exists" in registry
        assert "not_exists" not in registry
