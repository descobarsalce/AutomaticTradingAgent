"""
Feature selector that removes highly correlated features.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from src.data.feature_engineering.selectors.base_selector import BaseSelector

logger = logging.getLogger(__name__)


class CorrelationSelector(BaseSelector):
    """Remove highly correlated redundant features.

    When two features have correlation above the threshold,
    the one with lower variance is removed.
    """

    def __init__(self, threshold: float = 0.95):
        """Initialize correlation selector.

        Args:
            threshold: Correlation threshold above which to remove features
        """
        super().__init__()
        self.threshold = threshold

    def select(
        self,
        features: pd.DataFrame,
        target: Optional[pd.Series] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """Remove features with correlation above threshold.

        Args:
            features: DataFrame with feature columns
            target: Optional target variable (not used)
            top_k: Not used for correlation selector
            **kwargs: Additional arguments

        Returns:
            List of selected feature names
        """
        # Get numeric columns only
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return []

        numeric_features = features[numeric_cols]

        # Calculate correlation matrix
        corr_matrix = numeric_features.corr().abs()

        # Find pairs with high correlation
        upper_triangle = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        high_corr_pairs = np.where(
            (corr_matrix.values * upper_triangle) > self.threshold
        )

        # Determine which feature to remove from each pair
        to_remove = set()
        variances = numeric_features.var()

        for i, j in zip(*high_corr_pairs):
            col_i = corr_matrix.columns[i]
            col_j = corr_matrix.columns[j]

            # Skip if one is already marked for removal
            if col_i in to_remove or col_j in to_remove:
                continue

            # Keep the one with higher variance
            if variances[col_i] < variances[col_j]:
                to_remove.add(col_i)
            else:
                to_remove.add(col_j)

        # Return features not in removal set
        selected = [col for col in numeric_cols if col not in to_remove]

        logger.info(
            f"CorrelationSelector: Removed {len(to_remove)} highly correlated features"
        )

        return selected

    def get_correlation_report(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate a report of highly correlated feature pairs.

        Args:
            features: DataFrame with feature columns

        Returns:
            DataFrame with correlated pairs and their correlation values
        """
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = features[numeric_cols]

        corr_matrix = numeric_features.corr().abs()
        upper_triangle = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.threshold:
                    pairs.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j],
                    })

        return pd.DataFrame(pairs)

    def __repr__(self) -> str:
        return f"CorrelationSelector(threshold={self.threshold})"
