"""
Feature computation execution engine with parallel processing support.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.feature_engineering.sources.base_source import BaseFeatureSource
from src.data.feature_engineering.core.cache_manager import FeatureCacheManager

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Executes feature computations with optional parallelization and caching.

    Attributes:
        cache: Optional cache manager for computed features
        max_workers: Maximum parallel workers
        use_cache: Whether to use caching
    """

    def __init__(
        self,
        cache: Optional[FeatureCacheManager] = None,
        max_workers: int = 4,
        use_cache: bool = True,
    ):
        """Initialize execution engine.

        Args:
            cache: Optional cache manager
            max_workers: Maximum parallel workers
            use_cache: Whether to use caching
        """
        self.cache = cache
        self.max_workers = max_workers
        self.use_cache = use_cache and cache is not None

    def compute_features(
        self,
        sources: List[BaseFeatureSource],
        data: pd.DataFrame,
        symbols: List[str],
        feature_names: Optional[Dict[str, List[str]]] = None,
        parallel: bool = True,
    ) -> pd.DataFrame:
        """Compute features from multiple sources.

        Args:
            sources: List of feature sources to use
            data: Input DataFrame with OHLCV data
            symbols: List of symbols to compute features for
            feature_names: Optional dict mapping source names to feature lists
            parallel: Whether to compute in parallel

        Returns:
            DataFrame with all computed features
        """
        if parallel and len(sources) > 1:
            return self._compute_parallel(sources, data, symbols, feature_names)
        return self._compute_sequential(sources, data, symbols, feature_names)

    def _compute_sequential(
        self,
        sources: List[BaseFeatureSource],
        data: pd.DataFrame,
        symbols: List[str],
        feature_names: Optional[Dict[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """Compute features sequentially."""
        results = []

        for source in sources:
            if not source.enabled:
                continue

            try:
                features = feature_names.get(source.name) if feature_names else None
                result = self._compute_from_source(
                    source, data, symbols, features
                )
                if result is not None and not result.empty:
                    results.append(result)
            except Exception as e:
                logger.error(f"Source {source.name} failed: {e}")

        return self._merge_results(results, data.index)

    def _compute_parallel(
        self,
        sources: List[BaseFeatureSource],
        data: pd.DataFrame,
        symbols: List[str],
        feature_names: Optional[Dict[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """Compute features in parallel."""
        results = []
        enabled_sources = [s for s in sources if s.enabled]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_source = {}

            for source in enabled_sources:
                features = feature_names.get(source.name) if feature_names else None
                future = executor.submit(
                    self._compute_from_source,
                    source, data, symbols, features
                )
                future_to_source[future] = source.name

            for future in as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    result = future.result(timeout=30)
                    if result is not None and not result.empty:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Source {source_name} failed: {e}")

        return self._merge_results(results, data.index)

    def _compute_from_source(
        self,
        source: BaseFeatureSource,
        data: pd.DataFrame,
        symbols: List[str],
        features: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """Compute features from a single source, with caching."""
        # Check cache
        if self.use_cache and self.cache:
            cache_key = self.cache.generate_key(
                source.name,
                symbols,
                features or source.get_available_features(),
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {source.name}")
                # Filter to matching index
                common_idx = cached.index.intersection(data.index)
                return cached.loc[common_idx]

        # Validate data
        if not source.validate_data(data, symbols):
            logger.warning(f"Data validation failed for {source.name}")
            return None

        # Compute features
        result = source.compute_features(data, symbols, features)

        # Cache result
        if self.use_cache and self.cache and not result.empty:
            cache_key = self.cache.generate_key(
                source.name,
                symbols,
                features or source.get_available_features(),
            )
            self.cache.set(cache_key, result)

        return result

    def _merge_results(
        self,
        results: List[pd.DataFrame],
        index: pd.Index,
    ) -> pd.DataFrame:
        """Merge feature results from multiple sources."""
        if not results:
            return pd.DataFrame(index=index)

        # Concatenate horizontally
        merged = pd.concat(results, axis=1)

        # Remove duplicate columns if any
        merged = merged.loc[:, ~merged.columns.duplicated()]

        return merged

    def get_computation_stats(self) -> Dict[str, Any]:
        """Get computation statistics."""
        stats = {'cache_enabled': self.use_cache}
        if self.cache:
            stats['cache_stats'] = self.cache.get_stats()
        return stats

    def __repr__(self) -> str:
        return (
            f"ExecutionEngine(workers={self.max_workers}, "
            f"cache={self.use_cache})"
        )
