"""
Registry for managing prediction sources.

Handles registration, storage, and retrieval of prediction models.
"""
import logging
import os
from typing import Dict, List, Optional

from src.data.feature_engineering.predictions.base_prediction import BasePredictionSource

logger = logging.getLogger(__name__)


class PredictionRegistry:
    """Central registry for prediction sources.

    Manages:
    - Registration of prediction sources
    - Model persistence (save/load)
    - Source discovery and retrieval
    """

    def __init__(self, models_dir: str = "artifacts/models"):
        """Initialize registry.

        Args:
            models_dir: Directory for storing trained models
        """
        self._sources: Dict[str, BasePredictionSource] = {}
        self._models_dir = models_dir

        # Create models directory if it doesn't exist
        os.makedirs(self._models_dir, exist_ok=True)

    @property
    def models_dir(self) -> str:
        """Get models directory."""
        return self._models_dir

    def register_source(self, source: BasePredictionSource) -> None:
        """Register a prediction source.

        Args:
            source: Prediction source instance
        """
        if source.name in self._sources:
            logger.warning(f"Overwriting existing source: {source.name}")

        self._sources[source.name] = source
        logger.info(f"Registered prediction source: {source.name}")

        # Try to load existing model
        model_path = os.path.join(self._models_dir, source.name)
        if os.path.exists(model_path):
            if source.load_model(model_path):
                logger.info(f"Loaded existing model for {source.name}")

    def unregister_source(self, name: str) -> bool:
        """Unregister a prediction source.

        Args:
            name: Source name to unregister

        Returns:
            True if source was unregistered
        """
        if name in self._sources:
            del self._sources[name]
            logger.info(f"Unregistered prediction source: {name}")
            return True
        return False

    def get_source(self, name: str) -> Optional[BasePredictionSource]:
        """Get a registered source by name.

        Args:
            name: Source name

        Returns:
            Source instance or None
        """
        return self._sources.get(name)

    def get_all_sources(self) -> List[BasePredictionSource]:
        """Get all registered sources.

        Returns:
            List of all registered sources
        """
        return list(self._sources.values())

    def get_trained_sources(self) -> List[BasePredictionSource]:
        """Get all trained sources.

        Returns:
            List of sources that are trained
        """
        return [s for s in self._sources.values() if s.is_trained]

    def get_untrained_sources(self) -> List[BasePredictionSource]:
        """Get all untrained sources.

        Returns:
            List of sources that need training
        """
        return [s for s in self._sources.values() if not s.is_trained]

    def save_source_model(self, name: str) -> bool:
        """Save a source's model to disk.

        Args:
            name: Source name

        Returns:
            True if saved successfully
        """
        source = self._sources.get(name)
        if source is None:
            logger.error(f"Source not found: {name}")
            return False

        if not source.is_trained:
            logger.warning(f"Source {name} is not trained, skipping save")
            return False

        model_path = os.path.join(self._models_dir, name)
        os.makedirs(model_path, exist_ok=True)

        try:
            source.save_model(model_path)
            logger.info(f"Saved model for {name} to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model for {name}: {e}")
            return False

    def save_all_models(self) -> int:
        """Save all trained models to disk.

        Returns:
            Number of models saved
        """
        saved = 0
        for name in self._sources:
            if self.save_source_model(name):
                saved += 1
        return saved

    def load_source_model(self, name: str) -> bool:
        """Load a source's model from disk.

        Args:
            name: Source name

        Returns:
            True if loaded successfully
        """
        source = self._sources.get(name)
        if source is None:
            logger.error(f"Source not found: {name}")
            return False

        model_path = os.path.join(self._models_dir, name)
        if not os.path.exists(model_path):
            logger.warning(f"No saved model found for {name}")
            return False

        try:
            return source.load_model(model_path)
        except Exception as e:
            logger.error(f"Failed to load model for {name}: {e}")
            return False

    def get_all_feature_columns(self, symbols: List[str]) -> List[str]:
        """Get all feature columns from all sources.

        Args:
            symbols: List of symbols

        Returns:
            List of all feature column names
        """
        columns = []
        for source in self._sources.values():
            columns.extend(source.get_feature_columns(symbols))
        return columns

    def clear(self) -> None:
        """Clear all registered sources."""
        self._sources.clear()
        logger.info("Cleared all prediction sources")
