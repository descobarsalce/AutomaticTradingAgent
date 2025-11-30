"""
Abstract base class for all feature sources.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseFeatureSource(ABC):
    """Abstract base class for all feature sources.

    Every feature source plugin must implement this interface to be
    compatible with the feature engineering framework.

    Attributes:
        config: Configuration dictionary for the source
        name: Unique identifier for the source
        version: Semantic version string
        enabled: Whether the source is active
        priority: Priority for conflict resolution (higher wins)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature source.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        self.version = "1.0.0"
        self.enabled = True
        self.priority = 0

    @abstractmethod
    def get_available_features(self) -> List[str]:
        """Return list of features this source can provide.

        Returns:
            List of feature names (without symbol suffixes)
        """
        pass

    @abstractmethod
    def compute_features(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compute requested features for given symbols.

        Args:
            data: Input DataFrame with OHLCV data
            symbols: List of symbols to compute features for
            feature_names: Optional list of specific features to compute.
                          If None, compute all available features.

        Returns:
            DataFrame with computed features
        """
        pass

    def validate_data(self, data: pd.DataFrame, symbols: List[str]) -> bool:
        """Validate input data meets source requirements.

        Args:
            data: Input DataFrame to validate
            symbols: List of symbols to validate

        Returns:
            True if data is valid, False otherwise
        """
        if data is None or data.empty:
            logger.warning(f"{self.name}: Empty data provided")
            return False

        # Check for required columns based on dependencies
        for symbol in symbols:
            for dep in self.dependencies:
                col_name = f"{dep}_{symbol}"
                if col_name not in data.columns:
                    logger.warning(
                        f"{self.name}: Missing required column {col_name}"
                    )
                    return False

        return True

    @property
    def dependencies(self) -> List[str]:
        """List of required data column types (e.g., 'Close', 'Volume').

        Override in subclasses to specify required columns.

        Returns:
            List of column name prefixes this source requires
        """
        return []

    @property
    def metadata(self) -> Dict[str, Any]:
        """Source metadata for tracking and debugging.

        Returns:
            Dictionary with source information
        """
        return {
            'name': self.name,
            'version': self.version,
            'enabled': self.enabled,
            'priority': self.priority,
            'feature_count': len(self.get_available_features()),
            'dependencies': self.dependencies,
        }

    def __repr__(self) -> str:
        return (
            f"{self.name}(version={self.version}, "
            f"features={len(self.get_available_features())}, "
            f"priority={self.priority})"
        )
