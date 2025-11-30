"""Tests for the FeatureCacheManager."""

import pytest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

from data.feature_engineering.core.cache_manager import FeatureCacheManager


def create_sample_df() -> pd.DataFrame:
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
    })


class TestFeatureCacheManager:
    """Test suite for FeatureCacheManager."""

    def test_initialization(self):
        """Test cache manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCacheManager(cache_dir=tmpdir)

            assert cache.cache_dir == Path(tmpdir)
            assert cache.enable_disk is True
            assert len(cache.memory_cache) == 0

    def test_set_and_get_memory(self):
        """Test storing and retrieving from memory cache."""
        cache = FeatureCacheManager(enable_disk=False)
        df = create_sample_df()

        cache.set("test_key", df, persist=False)
        retrieved = cache.get("test_key")

        assert retrieved is not None
        pd.testing.assert_frame_equal(retrieved, df)

    def test_set_and_get_disk(self):
        """Test storing and retrieving from disk cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCacheManager(cache_dir=tmpdir)
            df = create_sample_df()

            cache.set("test_key", df)

            # Clear memory cache to force disk read
            cache.memory_cache.clear()

            retrieved = cache.get("test_key")
            assert retrieved is not None
            pd.testing.assert_frame_equal(retrieved, df)

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = FeatureCacheManager(enable_disk=False)

        result = cache.get("nonexistent_key")
        assert result is None

    def test_invalidate(self):
        """Test cache invalidation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCacheManager(cache_dir=tmpdir)
            df = create_sample_df()

            cache.set("test_key", df)
            assert cache.get("test_key") is not None

            cache.invalidate("test_key")
            assert cache.get("test_key") is None

    def test_clear(self):
        """Test clearing all cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCacheManager(cache_dir=tmpdir)
            df = create_sample_df()

            cache.set("key1", df)
            cache.set("key2", df)

            cache.clear()

            assert cache.get("key1") is None
            assert cache.get("key2") is None
            assert len(cache.memory_cache) == 0

    def test_generate_key(self):
        """Test cache key generation."""
        cache = FeatureCacheManager(enable_disk=False)

        key1 = cache.generate_key("source1", ["AAPL"], ["feat1"])
        key2 = cache.generate_key("source1", ["AAPL"], ["feat1"])
        key3 = cache.generate_key("source1", ["AAPL"], ["feat2"])

        # Same params should give same key
        assert key1 == key2
        # Different params should give different key
        assert key1 != key3

    def test_lru_eviction(self):
        """Test LRU eviction when memory is full."""
        cache = FeatureCacheManager(max_memory_items=3, enable_disk=False)

        # Add 4 items
        for i in range(4):
            cache.set(f"key{i}", create_sample_df(), persist=False)

        # First item should be evicted
        assert cache.get("key0") is None
        # Last 3 items should be present
        assert cache.get("key1") is not None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

    def test_get_stats(self):
        """Test statistics collection."""
        cache = FeatureCacheManager(enable_disk=False)
        df = create_sample_df()

        cache.set("hit_key", df, persist=False)

        # Generate some hits and misses
        cache.get("hit_key")  # hit
        cache.get("miss_key")  # miss
        cache.get("hit_key")  # hit

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3

    def test_returns_copy(self):
        """Test that get returns a copy, not the original."""
        cache = FeatureCacheManager(enable_disk=False)
        df = create_sample_df()

        cache.set("test_key", df, persist=False)
        retrieved = cache.get("test_key")

        # Modify retrieved DataFrame
        retrieved["feature1"] = 0

        # Original should be unchanged
        original = cache.get("test_key")
        assert not (original["feature1"] == 0).all()

    def test_repr(self):
        """Test string representation."""
        cache = FeatureCacheManager(enable_disk=False)
        repr_str = repr(cache)

        assert "FeatureCacheManager" in repr_str
        assert "memory" in repr_str
        assert "hit_rate" in repr_str
