"""
Base class for feature selection strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd


class BaseSelector(ABC):
    """Abstract base class for feature selectors.

    Feature selectors filter and rank features based on various criteria
    such as importance, correlation, or model-based metrics.
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize selector.

        Args:
            name: Optional name for the selector
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def select(
        self,
        features: pd.DataFrame,
        target: Optional[pd.Series] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """Select features based on the selector's criteria.

        Args:
            features: DataFrame with feature columns
            target: Optional target variable for supervised selection
            top_k: Number of features to select
            **kwargs: Additional selector-specific arguments

        Returns:
            List of selected feature names
        """
        pass

    def fit_transform(
        self,
        features: pd.DataFrame,
        target: Optional[pd.Series] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Select features and return filtered DataFrame.

        Args:
            features: DataFrame with feature columns
            target: Optional target variable for supervised selection
            top_k: Number of features to select
            **kwargs: Additional selector-specific arguments

        Returns:
            DataFrame with only selected features
        """
        selected = self.select(features, target, top_k, **kwargs)
        return features[selected]

    def __repr__(self) -> str:
        return f"{self.name}()"
