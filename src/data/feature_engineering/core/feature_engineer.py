"""
Main feature engineering orchestrator.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd
import yaml

from src.data.feature_engineering.core.feature_registry import FeatureRegistry
from src.data.feature_engineering.core.execution_engine import ExecutionEngine
from src.data.feature_engineering.core.cache_manager import FeatureCacheManager
from src.data.feature_engineering.sources.base_source import BaseFeatureSource

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Main orchestrator for feature computation.

    Coordinates feature sources, registry, caching, and execution
    to provide a unified interface for computing trading features.

    Example usage:
        engineer = FeatureEngineer()
        engineer.register_default_sources()
        features = engineer.compute_features(data, symbols=['AAPL', 'MSFT'])
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        enable_cache: bool = True,
        max_workers: int = 4,
    ):
        """Initialize feature engineer.

        Args:
            config: Configuration dictionary
            config_path: Path to YAML configuration file
            enable_cache: Whether to enable caching
            max_workers: Maximum parallel workers for computation
        """
        self.config = self._load_config(config, config_path)

        # Initialize components
        self.registry = FeatureRegistry()
        self.cache = FeatureCacheManager() if enable_cache else None
        self.engine = ExecutionEngine(
            cache=self.cache,
            max_workers=max_workers,
            use_cache=enable_cache,
        )

    def _load_config(
        self,
        config: Optional[Dict[str, Any]],
        config_path: Optional[str],
    ) -> Dict[str, Any]:
        """Load configuration from dict or file."""
        if config:
            return config

        if config_path:
            path = Path(config_path)
            if path.exists():
                with open(path) as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"Config file not found: {config_path}")

        return {}

    def register_source(self, source: BaseFeatureSource) -> None:
        """Register a feature source.

        Args:
            source: Feature source to register
        """
        self.registry.register_source(source)

    def register_source_class(
        self,
        source_class: Type[BaseFeatureSource],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a feature source by class.

        Args:
            source_class: Feature source class to instantiate and register
            config: Configuration for the source
        """
        source = source_class(config=config)
        self.register_source(source)

    def register_default_sources(self) -> None:
        """Register all default feature sources."""
        # Import here to avoid circular imports
        from src.data.feature_engineering.sources.market_data_source import MarketDataSource
        from src.data.feature_engineering.sources.technical_source import TechnicalSource

        self.register_source(MarketDataSource())
        self.register_source(TechnicalSource())

    def compute_features(
        self,
        data: pd.DataFrame,
        symbols: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        auto_select: bool = False,
        top_k: Optional[int] = None,
        drop_na: bool = True,
        current_index: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute features from registered sources.

        Args:
            data: Input DataFrame with OHLCV data
            symbols: List of symbols. If None, auto-detect from columns.
            sources: List of source names to use. If None, use all enabled.
            features: Specific features to compute. If None, compute all.
            auto_select: Whether to automatically select best features
            top_k: Number of features to select if auto_select is True
            drop_na: Whether to drop rows with NaN values
            current_index: If provided, only use data up to this index

        Returns:
            DataFrame with computed features
        """
        # Slice data if current_index provided
        if current_index is not None:
            data = data.iloc[:current_index + 1].copy()

        # Validate input
        if data is None or data.empty:
            raise ValueError("Input data is empty or None")

        # Auto-detect symbols if not provided
        if symbols is None:
            symbols = self._detect_symbols(data)

        if not symbols:
            raise ValueError("No symbols found in data")

        # Get sources to use
        if sources:
            source_list = [
                self.registry.get_source_by_name(s)
                for s in sources
                if self.registry.get_source_by_name(s)
            ]
        else:
            source_list = self.registry.get_enabled_sources()

        if not source_list:
            logger.warning("No sources available, returning original data")
            return data

        # Build feature name mapping if specific features requested
        feature_names = None
        if features:
            feature_names = self._map_features_to_sources(features)

        # Compute features
        result = self.engine.compute_features(
            sources=source_list,
            data=data,
            symbols=symbols,
            feature_names=feature_names,
        )

        # Merge with original data
        combined = pd.concat([data, result], axis=1)

        # Handle duplicate columns
        combined = combined.loc[:, ~combined.columns.duplicated()]

        # Feature selection
        if auto_select and top_k:
            combined = self._auto_select_features(combined, top_k)

        # Drop NaN rows
        if drop_na:
            combined = combined.dropna()

        return combined

    def _detect_symbols(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect symbols from column names."""
        symbols = set()
        for col in data.columns:
            if '_' in col:
                parts = col.split('_')
                if len(parts) >= 2:
                    # Assume format like 'Close_AAPL'
                    symbols.add(parts[-1])
        return sorted(list(symbols))

    def _map_features_to_sources(
        self,
        features: List[str],
    ) -> Dict[str, List[str]]:
        """Map requested features to their sources."""
        feature_names: Dict[str, List[str]] = {}

        for feature in features:
            source = self.registry.get_source(feature)
            if source:
                if source.name not in feature_names:
                    feature_names[source.name] = []
                feature_names[source.name].append(feature)

        return feature_names

    def _auto_select_features(
        self,
        data: pd.DataFrame,
        top_k: int,
    ) -> pd.DataFrame:
        """Select top K features based on variance."""
        # Simple variance-based selection for now
        # More sophisticated selectors can be plugged in
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        variances = data[numeric_cols].var()
        top_features = variances.nlargest(top_k).index.tolist()

        # Keep non-numeric columns plus top features
        non_numeric = [c for c in data.columns if c not in numeric_cols]
        return data[non_numeric + top_features]

    def list_features(self, source: Optional[str] = None) -> List[str]:
        """List all available features.

        Args:
            source: If provided, only list features from this source

        Returns:
            List of feature names
        """
        return self.registry.list_features(source)

    def list_sources(self) -> List[str]:
        """List all registered sources.

        Returns:
            List of source names
        """
        return self.registry.list_sources()

    def get_stats(self) -> Dict[str, Any]:
        """Get feature engineer statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'registry': self.registry.get_registry_stats(),
            'engine': self.engine.get_computation_stats(),
        }

    @classmethod
    def from_legacy(cls) -> 'FeatureEngineer':
        """Create feature engineer with legacy-compatible settings.

        This provides backward compatibility with the old FeatureEngineer.

        Returns:
            Configured FeatureEngineer instance
        """
        engineer = cls(enable_cache=False)
        engineer.register_default_sources()
        return engineer

    def prepare_data(
        self,
        portfolio_data: pd.DataFrame,
        current_index: Optional[int] = None,
    ) -> pd.DataFrame:
        """Legacy API: Prepare features from portfolio data.

        This method provides backward compatibility with the old API.

        Args:
            portfolio_data: Input DataFrame with OHLCV data
            current_index: If provided, only use data up to this index

        Returns:
            DataFrame with computed features
        """
        return self.compute_features(
            data=portfolio_data,
            current_index=current_index,
            drop_na=True,
        )

    def __repr__(self) -> str:
        return (
            f"FeatureEngineer(sources={len(self.registry.list_sources())}, "
            f"features={len(self.registry)})"
        )
