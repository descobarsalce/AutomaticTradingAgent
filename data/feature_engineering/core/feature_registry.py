"""
Central registry for all feature sources and their features.
"""

from typing import Any, Dict, List, Optional
import logging

from data.feature_engineering.sources.base_source import BaseFeatureSource

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """Central registry of all available features and their sources.

    The registry tracks which features are available and which source
    provides each feature. When multiple sources provide the same feature,
    the one with higher priority wins.

    Attributes:
        _sources: Dictionary of registered sources by name
        _feature_map: Mapping of feature names to source names
        _metadata: Cached metadata for each source
    """

    def __init__(self):
        """Initialize empty registry."""
        self._sources: Dict[str, BaseFeatureSource] = {}
        self._feature_map: Dict[str, str] = {}  # feature -> source name
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register_source(self, source: BaseFeatureSource) -> None:
        """Register a new feature source.

        If the source provides features that already exist from another source,
        the source with higher priority wins.

        Args:
            source: Feature source to register
        """
        self._sources[source.name] = source
        self._metadata[source.name] = source.metadata

        # Map features to source, handling conflicts via priority
        for feature in source.get_available_features():
            if feature in self._feature_map:
                existing_source = self._sources[self._feature_map[feature]]
                if source.priority > existing_source.priority:
                    logger.info(
                        f"Feature '{feature}': {source.name} (priority={source.priority}) "
                        f"overrides {existing_source.name} (priority={existing_source.priority})"
                    )
                    self._feature_map[feature] = source.name
            else:
                self._feature_map[feature] = source.name

        logger.info(
            f"Registered source: {source.name} with "
            f"{len(source.get_available_features())} features"
        )

    def unregister_source(self, source_name: str) -> bool:
        """Remove a source from the registry.

        Args:
            source_name: Name of source to remove

        Returns:
            True if source was removed, False if not found
        """
        if source_name not in self._sources:
            return False

        source = self._sources.pop(source_name)
        self._metadata.pop(source_name, None)

        # Remove feature mappings for this source
        features_to_remove = [
            f for f, s in self._feature_map.items() if s == source_name
        ]
        for feature in features_to_remove:
            del self._feature_map[feature]

        logger.info(f"Unregistered source: {source_name}")
        return True

    def get_source(self, feature_name: str) -> Optional[BaseFeatureSource]:
        """Get the source that provides a specific feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Source that provides the feature, or None if not found
        """
        source_name = self._feature_map.get(feature_name)
        return self._sources.get(source_name) if source_name else None

    def get_source_by_name(self, source_name: str) -> Optional[BaseFeatureSource]:
        """Get a source by its name.

        Args:
            source_name: Name of the source

        Returns:
            The source, or None if not found
        """
        return self._sources.get(source_name)

    def list_features(self, source_name: Optional[str] = None) -> List[str]:
        """List all available features.

        Args:
            source_name: If provided, only list features from this source

        Returns:
            List of feature names
        """
        if source_name:
            source = self._sources.get(source_name)
            return source.get_available_features() if source else []
        return list(self._feature_map.keys())

    def list_sources(self) -> List[str]:
        """List all registered source names.

        Returns:
            List of source names
        """
        return list(self._sources.keys())

    def get_all_sources(self) -> List[BaseFeatureSource]:
        """Get all registered sources.

        Returns:
            List of source instances
        """
        return list(self._sources.values())

    def get_enabled_sources(self) -> List[BaseFeatureSource]:
        """Get all enabled sources.

        Returns:
            List of enabled source instances
        """
        return [s for s in self._sources.values() if s.enabled]

    def get_feature_info(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Dictionary with feature information, or None if not found
        """
        source_name = self._feature_map.get(feature_name)
        if not source_name:
            return None

        source = self._sources[source_name]
        return {
            'feature': feature_name,
            'source': source_name,
            'source_version': source.version,
            'source_priority': source.priority,
        }

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the registry.

        Returns:
            Dictionary with registry statistics
        """
        return {
            'total_sources': len(self._sources),
            'enabled_sources': len(self.get_enabled_sources()),
            'total_features': len(self._feature_map),
            'sources': {
                name: len(source.get_available_features())
                for name, source in self._sources.items()
            },
        }

    def __len__(self) -> int:
        """Return number of registered features."""
        return len(self._feature_map)

    def __contains__(self, feature_name: str) -> bool:
        """Check if a feature is registered."""
        return feature_name in self._feature_map

    def __repr__(self) -> str:
        return (
            f"FeatureRegistry(sources={len(self._sources)}, "
            f"features={len(self._feature_map)})"
        )
