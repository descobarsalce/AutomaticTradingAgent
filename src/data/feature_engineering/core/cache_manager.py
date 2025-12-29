"""
Multi-level caching for computed features.
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureCacheManager:
    """Multi-level caching for computed features.

    Supports three cache levels:
    - L1: In-memory (fastest, limited size)
    - L2: Disk (parquet files)
    - L3: Distributed (Redis, optional)

    Attributes:
        memory_cache: In-memory cache dictionary
        cache_dir: Directory for disk cache
        max_memory_items: Maximum items in memory cache
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_memory_items: int = 100,
        enable_disk: bool = True,
    ):
        """Initialize cache manager.

        Args:
            cache_dir: Directory for disk cache. Defaults to .feature_cache
            max_memory_items: Maximum items to keep in memory
            enable_disk: Whether to enable disk caching
        """
        self.memory_cache: Dict[str, pd.DataFrame] = {}
        self.memory_access_order: List[str] = []
        self.max_memory_items = max_memory_items
        self.enable_disk = enable_disk

        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(".feature_cache")

        if self.enable_disk:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'disk_hits': 0,
        }

    def get(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get cached features.

        Tries memory cache first, then disk cache.

        Args:
            cache_key: Unique key for the cached data

        Returns:
            Cached DataFrame or None if not found
        """
        # Try L1: Memory
        if cache_key in self.memory_cache:
            self._stats['hits'] += 1
            self._stats['memory_hits'] += 1
            self._update_access_order(cache_key)
            return self.memory_cache[cache_key].copy()

        # Try L2: Disk
        if self.enable_disk:
            disk_path = self._get_disk_path(cache_key)
            if disk_path.exists():
                try:
                    data = pd.read_parquet(disk_path)
                    # Promote to L1
                    self._set_memory(cache_key, data)
                    self._stats['hits'] += 1
                    self._stats['disk_hits'] += 1
                    return data.copy()
                except Exception as e:
                    logger.warning(f"Failed to read cache file: {e}")

        self._stats['misses'] += 1
        return None

    def set(
        self,
        cache_key: str,
        data: pd.DataFrame,
        persist: bool = True,
    ) -> None:
        """Cache features.

        Args:
            cache_key: Unique key for the cached data
            data: DataFrame to cache
            persist: Whether to persist to disk
        """
        # L1: Memory
        self._set_memory(cache_key, data)

        # L2: Disk
        if persist and self.enable_disk:
            disk_path = self._get_disk_path(cache_key)
            try:
                data.to_parquet(disk_path)
            except Exception as e:
                logger.warning(f"Failed to write cache file: {e}")

    def invalidate(self, cache_key: str) -> bool:
        """Remove a specific cache entry.

        Args:
            cache_key: Key to invalidate

        Returns:
            True if entry was removed
        """
        removed = False

        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
            if cache_key in self.memory_access_order:
                self.memory_access_order.remove(cache_key)
            removed = True

        if self.enable_disk:
            disk_path = self._get_disk_path(cache_key)
            if disk_path.exists():
                disk_path.unlink()
                removed = True

        return removed

    def clear(self) -> None:
        """Clear all cache levels."""
        self.memory_cache.clear()
        self.memory_access_order.clear()

        if self.enable_disk and self.cache_dir.exists():
            for f in self.cache_dir.glob("*.parquet"):
                f.unlink()

        self._stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'disk_hits': 0,
        }

    def generate_key(
        self,
        source: str,
        symbols: List[str],
        features: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> str:
        """Generate a cache key from parameters.

        Args:
            source: Source name
            symbols: List of symbols
            features: List of feature names
            start_date: Start date of data
            end_date: End date of data

        Returns:
            MD5 hash key
        """
        key_parts = [
            source,
            "_".join(sorted(symbols)),
            "_".join(sorted(features)),
            str(start_date) if start_date else "",
            str(end_date) if end_date else "",
        ]
        key_data = "|".join(key_parts)
        return hashlib.md5(key_data.encode()).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = (
            self._stats['hits'] / total_requests if total_requests > 0 else 0
        )
        return {
            **self._stats,
            'hit_rate': hit_rate,
            'memory_size': len(self.memory_cache),
        }

    def _set_memory(self, cache_key: str, data: pd.DataFrame) -> None:
        """Set value in memory cache with LRU eviction."""
        # Evict if at capacity
        while len(self.memory_cache) >= self.max_memory_items:
            if self.memory_access_order:
                oldest = self.memory_access_order.pop(0)
                self.memory_cache.pop(oldest, None)

        self.memory_cache[cache_key] = data.copy()
        self._update_access_order(cache_key)

    def _update_access_order(self, cache_key: str) -> None:
        """Update LRU access order."""
        if cache_key in self.memory_access_order:
            self.memory_access_order.remove(cache_key)
        self.memory_access_order.append(cache_key)

    def _get_disk_path(self, cache_key: str) -> Path:
        """Get disk path for a cache key."""
        return self.cache_dir / f"{cache_key}.parquet"

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"FeatureCacheManager(memory={stats['memory_size']}, "
            f"hit_rate={stats['hit_rate']:.2%})"
        )
