"""
Feature selector based on importance scores.
"""

import logging
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from src.data.feature_engineering.selectors.base_selector import BaseSelector

logger = logging.getLogger(__name__)


class ImportanceSelector(BaseSelector):
    """Select features based on importance scores.

    Supports multiple importance methods:
    - variance: Select features with highest variance
    - mutual_info: Mutual information with target
    - f_score: F-statistic for regression
    - model_based: Use tree-based model feature importances
    """

    def __init__(
        self,
        method: Literal['variance', 'mutual_info', 'f_score', 'model_based'] = 'variance',
    ):
        """Initialize importance selector.

        Args:
            method: Importance calculation method
        """
        super().__init__()
        self.method = method

    def select(
        self,
        features: pd.DataFrame,
        target: Optional[pd.Series] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """Select top K features by importance.

        Args:
            features: DataFrame with feature columns
            target: Target variable (required for some methods)
            top_k: Number of features to select
            **kwargs: Additional arguments

        Returns:
            List of selected feature names
        """
        # Get numeric columns only
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return []

        numeric_features = features[numeric_cols]

        # Remove columns with zero variance
        variances = numeric_features.var()
        valid_cols = variances[variances > 0].index.tolist()
        numeric_features = numeric_features[valid_cols]

        if numeric_features.empty:
            return []

        # Calculate importance scores
        if self.method == 'variance':
            scores = numeric_features.var()
        elif self.method == 'mutual_info':
            scores = self._mutual_info_scores(numeric_features, target)
        elif self.method == 'f_score':
            scores = self._f_score(numeric_features, target)
        elif self.method == 'model_based':
            scores = self._model_importance(numeric_features, target)
        else:
            scores = numeric_features.var()

        # Select top K
        if top_k is None or top_k >= len(scores):
            top_k = len(scores)

        selected = scores.nlargest(top_k).index.tolist()
        return selected

    def _mutual_info_scores(
        self,
        features: pd.DataFrame,
        target: Optional[pd.Series],
    ) -> pd.Series:
        """Calculate mutual information scores."""
        if target is None:
            logger.warning("Target required for mutual_info, using variance")
            return features.var()

        try:
            from sklearn.feature_selection import mutual_info_regression

            # Align features and target
            common_idx = features.index.intersection(target.index)
            X = features.loc[common_idx].fillna(0)
            y = target.loc[common_idx].fillna(0)

            scores = mutual_info_regression(X, y)
            return pd.Series(scores, index=features.columns)
        except ImportError:
            logger.warning("sklearn not available, using variance")
            return features.var()

    def _f_score(
        self,
        features: pd.DataFrame,
        target: Optional[pd.Series],
    ) -> pd.Series:
        """Calculate F-statistic scores."""
        if target is None:
            logger.warning("Target required for f_score, using variance")
            return features.var()

        try:
            from sklearn.feature_selection import f_regression

            # Align features and target
            common_idx = features.index.intersection(target.index)
            X = features.loc[common_idx].fillna(0)
            y = target.loc[common_idx].fillna(0)

            scores, _ = f_regression(X, y)
            return pd.Series(scores, index=features.columns)
        except ImportError:
            logger.warning("sklearn not available, using variance")
            return features.var()

    def _model_importance(
        self,
        features: pd.DataFrame,
        target: Optional[pd.Series],
    ) -> pd.Series:
        """Calculate model-based feature importance."""
        if target is None:
            logger.warning("Target required for model_based, using variance")
            return features.var()

        try:
            from sklearn.ensemble import RandomForestRegressor

            # Align features and target
            common_idx = features.index.intersection(target.index)
            X = features.loc[common_idx].fillna(0)
            y = target.loc[common_idx].fillna(0)

            rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            rf.fit(X, y)

            return pd.Series(rf.feature_importances_, index=features.columns)
        except ImportError:
            logger.warning("sklearn not available, using variance")
            return features.var()

    def __repr__(self) -> str:
        return f"ImportanceSelector(method='{self.method}')"
