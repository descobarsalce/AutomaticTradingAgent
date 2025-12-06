"""
Walk-forward validation for time series prediction models.

Implements rolling window validation to simulate real trading conditions
and provide realistic out-of-sample performance metrics.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.feature_engineering.predictions.base_prediction import (
    BasePredictionSource,
    PredictionMode,
)

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results from a single validation fold."""
    fold_idx: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_start_date: Optional[str] = None
    train_end_date: Optional[str] = None
    test_start_date: Optional[str] = None
    test_end_date: Optional[str] = None

    # Metrics per horizon
    mae_by_horizon: Dict[int, float] = field(default_factory=dict)
    rmse_by_horizon: Dict[int, float] = field(default_factory=dict)
    direction_accuracy_by_horizon: Dict[int, float] = field(default_factory=dict)

    # Raw predictions and actuals for plotting
    predictions: Dict[int, List[float]] = field(default_factory=dict)
    actuals: Dict[int, List[float]] = field(default_factory=dict)


@dataclass
class WalkForwardResults:
    """Aggregated results from walk-forward validation."""
    symbol: str
    model_name: str
    num_folds: int

    # Aggregated metrics (mean across folds)
    mae_by_horizon: Dict[int, float] = field(default_factory=dict)
    rmse_by_horizon: Dict[int, float] = field(default_factory=dict)
    direction_accuracy_by_horizon: Dict[int, float] = field(default_factory=dict)

    # Std of metrics across folds
    mae_std_by_horizon: Dict[int, float] = field(default_factory=dict)
    rmse_std_by_horizon: Dict[int, float] = field(default_factory=dict)
    direction_accuracy_std_by_horizon: Dict[int, float] = field(default_factory=dict)

    # Per-fold details
    fold_results: List[FoldResult] = field(default_factory=list)

    # All out-of-sample predictions (for plotting)
    all_predictions: Dict[int, List[float]] = field(default_factory=dict)
    all_actuals: Dict[int, List[float]] = field(default_factory=dict)
    all_dates: List[str] = field(default_factory=list)

    # Training stats
    total_folds_attempted: int = 0
    training_successes: int = 0
    training_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display/storage."""
        return {
            'symbol': self.symbol,
            'model_name': self.model_name,
            'num_folds': self.num_folds,
            'metrics': {
                horizon: {
                    'mae': self.mae_by_horizon.get(horizon, 0),
                    'mae_std': self.mae_std_by_horizon.get(horizon, 0),
                    'rmse': self.rmse_by_horizon.get(horizon, 0),
                    'rmse_std': self.rmse_std_by_horizon.get(horizon, 0),
                    'direction_accuracy': self.direction_accuracy_by_horizon.get(horizon, 0),
                    'direction_accuracy_std': self.direction_accuracy_std_by_horizon.get(horizon, 0),
                }
                for horizon in self.mae_by_horizon.keys()
            },
            'fold_results': [
                {
                    'fold': f.fold_idx,
                    'train_period': f"{f.train_start_date} to {f.train_end_date}",
                    'test_period': f"{f.test_start_date} to {f.test_end_date}",
                    'mae': f.mae_by_horizon,
                    'direction_accuracy': f.direction_accuracy_by_horizon,
                }
                for f in self.fold_results
            ]
        }


class WalkForwardValidator:
    """Performs walk-forward validation for time series prediction models.

    Walk-forward validation trains on a rolling window and tests on the
    subsequent period, simulating how the model would perform in real trading.

    Example:
        validator = WalkForwardValidator(
            train_window_days=252,  # 1 year training
            test_window_days=21,    # 1 month testing
            step_days=21,           # Step forward monthly
        )
        results = validator.validate(lstm_predictor, data, 'AAPL')
    """

    def __init__(
        self,
        train_window_days: int = 252,
        test_window_days: int = 21,
        step_days: int = 21,
    ):
        """Initialize walk-forward validator.

        Args:
            train_window_days: Number of days for training window
            test_window_days: Number of days for testing window
            step_days: Number of days to step forward between folds
        """
        self._train_window = train_window_days
        self._test_window = test_window_days
        self._step = step_days

    @property
    def train_window_days(self) -> int:
        return self._train_window

    @train_window_days.setter
    def train_window_days(self, value: int) -> None:
        self._train_window = max(30, value)

    @property
    def test_window_days(self) -> int:
        return self._test_window

    @test_window_days.setter
    def test_window_days(self, value: int) -> None:
        self._test_window = max(5, value)

    @property
    def step_days(self) -> int:
        return self._step

    @step_days.setter
    def step_days(self, value: int) -> None:
        self._step = max(1, value)

    def _generate_folds(
        self,
        data_length: int,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> List[Tuple[int, int, int, int]]:
        """Generate train/test fold indices.

        Args:
            data_length: Total length of data
            start_idx: Starting index for validation
            end_idx: Ending index for validation

        Returns:
            List of tuples: (train_start, train_end, test_start, test_end)
        """
        if end_idx is None:
            end_idx = data_length

        folds = []
        current_train_start = start_idx

        while True:
            train_end = current_train_start + self._train_window
            test_start = train_end
            test_end = test_start + self._test_window

            # Stop if test window exceeds data
            if test_end > end_idx:
                break

            folds.append((current_train_start, train_end, test_start, test_end))
            current_train_start += self._step

        return folds

    def validate(
        self,
        predictor: BasePredictionSource,
        data: pd.DataFrame,
        symbol: str,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> WalkForwardResults:
        """Run walk-forward validation.

        Args:
            predictor: Prediction model to validate
            data: Full DataFrame with OHLCV data
            symbol: Symbol to validate on
            start_idx: Starting index for validation
            end_idx: Ending index for validation
            progress_callback: Optional callback(fold_idx, total_folds) for progress

        Returns:
            WalkForwardResults with aggregated and per-fold metrics
        """
        if end_idx is None:
            end_idx = len(data)

        folds = self._generate_folds(len(data), start_idx, end_idx)

        if not folds:
            logger.warning("Insufficient data for walk-forward validation")
            return WalkForwardResults(
                symbol=symbol,
                model_name=predictor.name,
                num_folds=0,
            )

        logger.info(
            f"Running walk-forward validation with {len(folds)} folds "
            f"for {predictor.name} on {symbol}"
        )

        fold_results = []
        all_predictions: Dict[int, List[float]] = {h: [] for h in predictor.horizons}
        all_actuals: Dict[int, List[float]] = {h: [] for h in predictor.horizons}
        all_dates: List[str] = []

        training_failures = 0
        training_successes = 0

        for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
            if progress_callback:
                progress_callback(fold_idx, len(folds))

            logger.debug(
                f"Fold {fold_idx + 1}/{len(folds)}: "
                f"train [{train_start}:{train_end}], test [{test_start}:{test_end}]"
            )

            # Train on this fold
            try:
                # Reset trained state before training
                predictor._is_trained = False
                logger.info(f"Starting training for fold {fold_idx}: [{train_start}:{train_end}]")
                predictor.train(data, [symbol], train_start, train_end)

                if not predictor.is_trained:
                    training_failures += 1
                    logger.warning(
                        f"Fold {fold_idx}: Training completed but model not marked as trained. "
                        f"This usually means insufficient training samples were generated. "
                        f"Check data columns and index ranges."
                    )
                    continue
                else:
                    training_successes += 1
                    logger.info(f"Fold {fold_idx}: Training succeeded")
            except Exception as e:
                training_failures += 1
                logger.error(f"Training failed on fold {fold_idx} with error: {type(e).__name__}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

            # Collect predictions and actuals for test window
            fold_predictions: Dict[int, List[float]] = {h: [] for h in predictor.horizons}
            fold_actuals: Dict[int, List[float]] = {h: [] for h in predictor.horizons}
            fold_dates: List[str] = []

            for test_idx in range(test_start, test_end):
                try:
                    predictions = predictor.predict(data, [symbol], test_idx)

                    if symbol in predictions:
                        for horizon in predictor.horizons:
                            if horizon in predictions[symbol]:
                                pred_value = predictions[symbol][horizon].get(
                                    PredictionMode.PRICE, 0.0
                                )

                                # Compute actual target
                                targets = predictor._compute_targets(
                                    data, symbol, test_idx, horizon
                                )
                                if targets is not None:
                                    actual_value = targets[PredictionMode.PRICE]

                                    fold_predictions[horizon].append(pred_value)
                                    fold_actuals[horizon].append(actual_value)

                    # Store date for this prediction
                    if hasattr(data.index, 'strftime'):
                        fold_dates.append(str(data.index[test_idx]))
                    else:
                        fold_dates.append(str(test_idx))

                except Exception as e:
                    logger.debug(f"Prediction failed at idx {test_idx}: {e}")
                    continue

            # Compute metrics for this fold
            fold_result = self._compute_fold_metrics(
                fold_idx=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                data=data,
                horizons=predictor.horizons,
                predictions=fold_predictions,
                actuals=fold_actuals,
            )
            fold_results.append(fold_result)

            # Accumulate for overall metrics
            for horizon in predictor.horizons:
                all_predictions[horizon].extend(fold_predictions[horizon])
                all_actuals[horizon].extend(fold_actuals[horizon])
            all_dates.extend(fold_dates)

        # Aggregate metrics across folds
        results = self._aggregate_results(
            symbol=symbol,
            model_name=predictor.name,
            horizons=predictor.horizons,
            fold_results=fold_results,
            all_predictions=all_predictions,
            all_actuals=all_actuals,
            all_dates=all_dates,
        )

        # Add training stats
        results.total_folds_attempted = len(folds)
        results.training_successes = training_successes
        results.training_failures = training_failures

        logger.info(
            f"Walk-forward validation complete: {len(fold_results)} successful folds out of {len(folds)} attempted. "
            f"Training successes: {training_successes}, failures: {training_failures}"
        )
        if len(fold_results) > 0:
            logger.info(f"MAE={results.mae_by_horizon.get(1, 0):.4f} (1d)")

        return results

    def _compute_fold_metrics(
        self,
        fold_idx: int,
        train_start: int,
        train_end: int,
        test_start: int,
        test_end: int,
        data: pd.DataFrame,
        horizons: List[int],
        predictions: Dict[int, List[float]],
        actuals: Dict[int, List[float]],
    ) -> FoldResult:
        """Compute metrics for a single fold."""
        # Get date strings if available
        try:
            train_start_date = str(data.index[train_start])[:10]
            train_end_date = str(data.index[train_end - 1])[:10]
            test_start_date = str(data.index[test_start])[:10]
            test_end_date = str(data.index[test_end - 1])[:10]
        except (IndexError, TypeError):
            train_start_date = str(train_start)
            train_end_date = str(train_end)
            test_start_date = str(test_start)
            test_end_date = str(test_end)

        result = FoldResult(
            fold_idx=fold_idx,
            train_start_idx=train_start,
            train_end_idx=train_end,
            test_start_idx=test_start,
            test_end_idx=test_end,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
            predictions=predictions,
            actuals=actuals,
        )

        for horizon in horizons:
            preds = np.array(predictions.get(horizon, []))
            acts = np.array(actuals.get(horizon, []))

            if len(preds) > 0 and len(acts) > 0:
                # MAE
                result.mae_by_horizon[horizon] = float(np.mean(np.abs(preds - acts)))

                # RMSE
                result.rmse_by_horizon[horizon] = float(
                    np.sqrt(np.mean((preds - acts) ** 2))
                )

                # Direction accuracy
                pred_direction = np.sign(preds)
                actual_direction = np.sign(acts)
                result.direction_accuracy_by_horizon[horizon] = float(
                    np.mean(pred_direction == actual_direction)
                )
            else:
                result.mae_by_horizon[horizon] = 0.0
                result.rmse_by_horizon[horizon] = 0.0
                result.direction_accuracy_by_horizon[horizon] = 0.0

        return result

    def _aggregate_results(
        self,
        symbol: str,
        model_name: str,
        horizons: List[int],
        fold_results: List[FoldResult],
        all_predictions: Dict[int, List[float]],
        all_actuals: Dict[int, List[float]],
        all_dates: List[str],
    ) -> WalkForwardResults:
        """Aggregate metrics across all folds."""
        results = WalkForwardResults(
            symbol=symbol,
            model_name=model_name,
            num_folds=len(fold_results),
            fold_results=fold_results,
            all_predictions=all_predictions,
            all_actuals=all_actuals,
            all_dates=all_dates,
        )

        for horizon in horizons:
            # Collect metrics from all folds
            maes = [f.mae_by_horizon.get(horizon, 0) for f in fold_results if horizon in f.mae_by_horizon]
            rmses = [f.rmse_by_horizon.get(horizon, 0) for f in fold_results if horizon in f.rmse_by_horizon]
            dir_accs = [f.direction_accuracy_by_horizon.get(horizon, 0) for f in fold_results if horizon in f.direction_accuracy_by_horizon]

            if maes:
                results.mae_by_horizon[horizon] = float(np.mean(maes))
                results.mae_std_by_horizon[horizon] = float(np.std(maes))

            if rmses:
                results.rmse_by_horizon[horizon] = float(np.mean(rmses))
                results.rmse_std_by_horizon[horizon] = float(np.std(rmses))

            if dir_accs:
                results.direction_accuracy_by_horizon[horizon] = float(np.mean(dir_accs))
                results.direction_accuracy_std_by_horizon[horizon] = float(np.std(dir_accs))

        return results
