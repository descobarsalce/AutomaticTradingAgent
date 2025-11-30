"""
Enhanced Hyperparameter Search Module
Two-phase Optuna optimization: broad exploration → focused exploitation
"""
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import optuna
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler

from core.base_agent import UnifiedTradingAgent

logger = logging.getLogger(__name__)


@dataclass
class ParamRange:
    """Parameter range specification with type information."""

    low: float
    high: float
    param_type: str  # 'float', 'int', 'loguniform'
    step: Optional[float] = None


class TwoPhaseHyperparameterOptimizer:
    """
    Advanced two-phase hyperparameter optimization using Optuna.

    Phase 1: Broad exploration of the parameter space
    Phase 2: Focused exploitation of promising regions
    """

    # Configuration constants
    TOP_FRAC = 0.20  # Keep best 20% from Phase 1
    MARGIN_FACTOR = 0.10  # ±10% padding around min-max for Phase 2

    def __init__(
        self,
        stock_names: list,
        train_start_date: datetime,
        train_end_date: datetime,
        env_params: Dict[str, Any],
        param_ranges: Dict[str, Tuple[float, float]],
        optimization_metric: str = "sharpe_ratio",
        progress_bar=None,
        status_text=None,
    ):
        self.stock_names = stock_names
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.env_params = env_params
        self.param_ranges = param_ranges
        self.optimization_metric = optimization_metric
        self.progress_bar = progress_bar
        self.status_text = status_text

        # Results storage
        self.phase1_study: Optional[optuna.Study] = None
        self.phase2_study: Optional[optuna.Study] = None
        self.all_trials: list = []
        self.refined_ranges: Dict[str, Tuple[float, float]] = {}

        # Phase accounting
        self.phase1_trials_count: int = 0
        self.phase2_trials_count: int = 0
        self.phase1_ratio: float = 0.0

        # Trial counter for progress
        self.total_trials = 0
        self.completed_trials = 0

        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def run_optimization(
        self,
        n_trials: int,
        pruning_enabled: bool = True,
        phase1_ratio: float = 0.6,
        iterative: bool = False,
        max_rounds: int = 3,
        improvement_threshold: float = 0.01,
    ) -> optuna.Study:
        """
        Run complete two-phase optimization.

        Args:
            n_trials: Trials per iteration across both phases
            pruning_enabled: Whether to enable early trial pruning
            phase1_ratio: Fraction of trials for Phase 1 (default 60%)
            iterative: Whether to repeat two-phase cycles until improvement stalls
            max_rounds: Maximum refinement cycles to run
            improvement_threshold: Minimum relative improvement needed to keep refining

        Returns:
            The best study (from whichever phase performed better)
        """
        if iterative:
            return self._run_iterative_rounds(
                n_trials=n_trials,
                pruning_enabled=pruning_enabled,
                phase1_ratio=phase1_ratio,
                max_rounds=max_rounds,
                improvement_threshold=improvement_threshold,
            )

        return self._run_single_round(
            n_trials=n_trials, pruning_enabled=pruning_enabled, phase1_ratio=phase1_ratio
        )

    def _run_single_round(
        self, n_trials: int, pruning_enabled: bool, phase1_ratio: float
    ) -> optuna.Study:
        """Run one exploration→exploitation cycle."""
        phase1_trials = int(n_trials * phase1_ratio)
        phase2_trials = n_trials - phase1_trials
        self.phase1_trials_count = phase1_trials
        self.phase2_trials_count = phase2_trials
        self.phase1_ratio = phase1_ratio
        self.total_trials = n_trials
        self.completed_trials = 0
        self.round_studies = []
        self.round_summaries = []

        if self.status_text:
            self.status_text.text(f"Phase 1: Broad exploration ({phase1_trials} trials)")

        # Phase 1: Exploration
        self.phase1_study = self._run_phase1(phase1_trials, pruning_enabled)
        self.round_studies.append(
            {
                "round": 1,
                "phase": 1,
                "study": self.phase1_study,
                "trial_offset": 0,
            }
        )

        # Extract promising regions
        self.refined_ranges = self._extract_promising_regions()

        if self.status_text:
            self.status_text.text(f"Phase 2: Focused exploitation ({phase2_trials} trials)")

        # Phase 2: Exploitation
        self.phase2_study = self._run_phase2(phase2_trials, pruning_enabled)
        self.round_studies.append(
            {
                "round": 1,
                "phase": 2,
                "study": self.phase2_study,
                "trial_offset": len(self.phase1_study.trials),
            }
        )

        best_study = self._select_best_study()
        self.best_phase_overall = 1 if best_study == self.phase1_study else 2
        self.round_summaries.append(
            {
                "round": 1,
                "phase1_trials": phase1_trials,
                "phase2_trials": phase2_trials,
                "best_value": best_study.best_value if best_study and best_study.best_trial else float("-inf"),
                "refined_ranges": self.refined_ranges,
            }
        )
        return best_study

    def _run_iterative_rounds(
        self,
        n_trials: int,
        pruning_enabled: bool,
        phase1_ratio: float,
        max_rounds: int,
        improvement_threshold: float,
    ) -> optuna.Study:
        """Run iterative exploration→exploitation cycles until improvement stalls."""
        self.round_studies = []
        self.round_summaries = []
        self.total_trials = n_trials * max_rounds
        self.completed_trials = 0
        current_ranges = self.param_ranges
        previous_best = float("-inf")
        best_study_overall: Optional[optuna.Study] = None
        best_phase_overall = None
        trial_offset = 0

        for current_round in range(1, max_rounds + 1):
            phase1_trials = int(n_trials * phase1_ratio)
            phase2_trials = n_trials - phase1_trials
            self.phase1_trials_count = phase1_trials
            self.phase2_trials_count = phase2_trials
            self.phase1_ratio = phase1_ratio
            self.param_ranges = current_ranges

            if self.status_text:
                self.status_text.text(
                    f"Round {current_round}: Phase 1 exploration ({phase1_trials} trials)"
                )

            self.phase1_study = self._run_phase1(phase1_trials, pruning_enabled)
            self.round_studies.append(
                {
                    "round": current_round,
                    "phase": 1,
                    "study": self.phase1_study,
                    "trial_offset": trial_offset,
                }
            )
            trial_offset += len(self.phase1_study.trials)

            self.refined_ranges = self._extract_promising_regions()
            if self.status_text:
                self.status_text.text(
                    f"Round {current_round}: Phase 2 exploitation ({phase2_trials} trials)"
                )

            self.phase2_study = self._run_phase2(phase2_trials, pruning_enabled)
            self.round_studies.append(
                {
                    "round": current_round,
                    "phase": 2,
                    "study": self.phase2_study,
                    "trial_offset": trial_offset,
                }
            )
            trial_offset += len(self.phase2_study.trials)

            round_best_study = self._select_best_study()
            round_best_value = (
                round_best_study.best_value
                if round_best_study and round_best_study.best_trial
                else float("-inf")
            )

            improvement = (
                (round_best_value - previous_best) / abs(previous_best)
                if previous_best not in (-float("inf"), float("-inf")) and previous_best != 0
                else float("inf")
            )

            self.round_summaries.append(
                {
                    "round": current_round,
                    "phase1_trials": phase1_trials,
                    "phase2_trials": phase2_trials,
                    "best_value": round_best_value,
                    "improvement": improvement,
                    "refined_ranges": self.refined_ranges,
                }
            )

            if (
                best_study_overall is None
                or (round_best_study and round_best_value > best_study_overall.best_value)
            ):
                best_study_overall = round_best_study
                best_phase_overall = 1 if round_best_study == self.phase1_study else 2

            if previous_best != float("-inf") and improvement < improvement_threshold:
                logger.info(
                    "Stopping iterative refinement: improvement %.4f below threshold %.4f",
                    improvement,
                    improvement_threshold,
                )
                break

            previous_best = max(previous_best, round_best_value)
            current_ranges = self.refined_ranges if self.refined_ranges else current_ranges

        if best_study_overall is None:
            best_study_overall = self.phase1_study or self.phase2_study
        self.best_phase_overall = best_phase_overall
        return best_study_overall

    def _run_phase1(self, n_trials: int, pruning_enabled: bool) -> optuna.Study:
        """Phase 1: Broad exploration of parameter space."""

        # Configure sampler for exploration (more random sampling initially)
        explore_sampler = TPESampler(
            multivariate=True,
            constant_liar=True,
            n_startup_trials=max(10, n_trials // 5),  # More random trials initially
            n_ei_candidates=24,
            seed=42,
        )

        # Configure pruner
        pruner = self._get_pruner(pruning_enabled, phase=1)

        study = optuna.create_study(
            direction="maximize",
            sampler=explore_sampler,
            pruner=pruner,
            study_name="ppo_phase1_exploration",
        )

        study.optimize(
            lambda trial: self._objective(trial, self.param_ranges, phase=1),
            n_trials=n_trials,
            callbacks=[self._trial_callback],
            show_progress_bar=False,
        )

        return study

    def _run_phase2(self, n_trials: int, pruning_enabled: bool) -> optuna.Study:
        """Phase 2: Focused exploitation of promising regions."""

        # Use refined ranges if available, otherwise fall back to original
        ranges_to_use = self.refined_ranges if self.refined_ranges else self.param_ranges

        # Configure sampler for exploitation (less exploration)
        exploit_sampler = TPESampler(
            multivariate=True,
            constant_liar=True,
            n_startup_trials=max(5, n_trials // 10),  # Fewer random trials
            n_ei_candidates=16,
            seed=43,
            consider_prior=True,
        )

        pruner = self._get_pruner(pruning_enabled, phase=2)

        study = optuna.create_study(
            direction="maximize",
            sampler=exploit_sampler,
            pruner=pruner,
            study_name="ppo_phase2_exploitation",
        )

        study.optimize(
            lambda trial: self._objective(trial, ranges_to_use, phase=2),
            n_trials=n_trials,
            callbacks=[self._trial_callback],
            show_progress_bar=False,
        )

        return study

    def _get_pruner(self, enabled: bool, phase: int):
        """Get appropriate pruner for the phase."""
        if not enabled:
            return None

        if phase == 1:
            # More aggressive pruning in exploration phase
            return SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)
        else:
            # Gentler pruning in exploitation phase
            return MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    def _objective(
        self, trial: optuna.Trial, param_ranges: Dict[str, Tuple[float, float]], phase: int
    ) -> float:
        """Objective function for hyperparameter optimization."""
        try:
            # Sample PPO parameters
            ppo_params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", param_ranges["lr"][0], param_ranges["lr"][1], log=True
                ),
                "n_steps": trial.suggest_int(
                    "n_steps", int(param_ranges["steps"][0]), int(param_ranges["steps"][1]), step=64
                ),
                "batch_size": trial.suggest_int(
                    "batch_size", int(param_ranges["batch"][0]), int(param_ranges["batch"][1]), step=32
                ),
                "n_epochs": trial.suggest_int(
                    "n_epochs", int(param_ranges["epochs"][0]), int(param_ranges["epochs"][1])
                ),
                "gamma": trial.suggest_float(
                    "gamma", param_ranges["gamma"][0], param_ranges["gamma"][1]
                ),
                "gae_lambda": trial.suggest_float(
                    "gae_lambda", param_ranges["gae"][0], param_ranges["gae"][1]
                ),
            }

            # Update status
            if self.status_text:
                phase_str = "Exploration" if phase == 1 else "Exploitation"
                self.status_text.text(
                    f"Phase {phase} ({phase_str}) - Trial {trial.number + 1}: "
                    f"lr={ppo_params['learning_rate']:.2e}, "
                    f"steps={ppo_params['n_steps']}, "
                    f"gamma={ppo_params['gamma']:.3f}"
                )

            # Train model
            trial_model = UnifiedTradingAgent()
            metrics = trial_model.train(
                stock_names=self.stock_names,
                start_date=self.train_start_date,
                end_date=self.train_end_date,
                env_params=self.env_params,
                ppo_params=ppo_params,
            )

            trial_value = metrics.get(self.optimization_metric, float("-inf"))

            # Store trial info
            self.all_trials.append(
                {
                    "trial_number": len(self.all_trials),
                    "phase": phase,
                    "params": ppo_params,
                    "value": trial_value,
                    "state": "COMPLETE",
                }
            )

            return trial_value

        except optuna.TrialPruned:
            self.all_trials.append(
                {
                    "trial_number": len(self.all_trials),
                    "phase": phase,
                    "params": trial.params,
                    "value": None,
                    "state": "PRUNED",
                }
            )
            raise

        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            self.all_trials.append(
                {
                    "trial_number": len(self.all_trials),
                    "phase": phase,
                    "params": trial.params if hasattr(trial, "params") else {},
                    "value": float("-inf"),
                    "state": "FAILED",
                }
            )
            return float("-inf")

    def _trial_callback(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """Callback for progress tracking."""
        self.completed_trials += 1
        if self.progress_bar:
            self.progress_bar.progress(self.completed_trials / self.total_trials)

    def _extract_promising_regions(self) -> Dict[str, Tuple[float, float]]:
        """Extract promising parameter regions from Phase 1 results."""
        if not self.phase1_study or not self.phase1_study.trials:
            return self.param_ranges.copy()

        # Get completed trials
        completed_trials = [
            t
            for t in self.phase1_study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]

        if len(completed_trials) < 3:
            logger.warning("Too few completed trials in Phase 1, using original ranges")
            return self.param_ranges.copy()

        # Select top performers
        n_best = max(1, int(len(completed_trials) * self.TOP_FRAC))
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
        best_trials = sorted_trials[:n_best]

        logger.info(
            f"Extracting regions from top {n_best} trials "
            f"(best value: {best_trials[0].value:.4f})"
        )

        # Extract bounds for each parameter
        refined_ranges = {}
        param_mapping = {
            "learning_rate": "lr",
            "n_steps": "steps",
            "batch_size": "batch",
            "n_epochs": "epochs",
            "gamma": "gamma",
            "gae_lambda": "gae",
        }

        for param_name, range_key in param_mapping.items():
            values = [t.params.get(param_name) for t in best_trials if param_name in t.params]

            if not values:
                refined_ranges[range_key] = self.param_ranges[range_key]
                continue

            orig_low, orig_high = self.param_ranges[range_key]
            val_min, val_max = min(values), max(values)

            # Add padding
            padding = (val_max - val_min) * self.MARGIN_FACTOR
            if padding == 0:
                padding = abs(val_min) * self.MARGIN_FACTOR or 1e-6

            # Apply bounds with original constraints
            new_low = max(orig_low, val_min - padding)
            new_high = min(orig_high, val_max + padding)

            # Ensure we have a valid range
            if new_low >= new_high:
                new_low = val_min * 0.9
                new_high = val_max * 1.1

            refined_ranges[range_key] = (new_low, new_high)

            logger.info(
                f"  {range_key}: [{orig_low:.4g}, {orig_high:.4g}] "
                f"→ [{new_low:.4g}, {new_high:.4g}]"
            )

        return refined_ranges

    def _select_best_study(self) -> optuna.Study:
        """Select the study with the best result."""
        phase1_best = (
            self.phase1_study.best_value
            if self.phase1_study and self.phase1_study.best_trial
            else float("-inf")
        )
        phase2_best = (
            self.phase2_study.best_value
            if self.phase2_study and self.phase2_study.best_trial
            else float("-inf")
        )

        if phase1_best >= phase2_best:
            logger.info(f"Phase 1 produced best result: {phase1_best:.4f}")
            return self.phase1_study
        else:
            logger.info(f"Phase 2 produced best result: {phase2_best:.4f}")
            return self.phase2_study

    def get_combined_study_data(self) -> pd.DataFrame:
        """Get combined trial data from all rounds and phases."""
        trials_data = []

        if hasattr(self, "round_studies") and self.round_studies:
            for phase_record in self.round_studies:
                study = phase_record["study"]
                if not study:
                    continue
                for t in study.trials:
                    if t.value is not None:
                        trials_data.append(
                            {
                                "Trial": t.number + phase_record.get("trial_offset", 0),
                                "Round": phase_record.get("round"),
                                "Phase": phase_record.get("phase"),
                                "Value": t.value,
                                "State": str(t.state),
                                **t.params,
                            }
                        )
        else:
            # Fallback to the latest two-phase run
            if self.phase1_study:
                for t in self.phase1_study.trials:
                    if t.value is not None:
                        trials_data.append(
                            {
                                "Trial": t.number,
                                "Phase": 1,
                                "Value": t.value,
                                "State": str(t.state),
                                **t.params,
                            }
                        )

            phase1_count = len(self.phase1_study.trials) if self.phase1_study else 0
            if self.phase2_study:
                for t in self.phase2_study.trials:
                    if t.value is not None:
                        trials_data.append(
                            {
                                "Trial": t.number + phase1_count,
                                "Phase": 2,
                                "Value": t.value,
                                "State": str(t.state),
                                **t.params,
                            }
                        )

        return pd.DataFrame(trials_data)


def create_parameter_ranges() -> Dict[str, Tuple[float, float]]:
    """Get parameter ranges from user input."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Core Parameters")
        lr_min = st.number_input("Learning Rate Min", value=1e-5, format="%.1e")
        lr_max = st.number_input("Learning Rate Max", value=5e-4, format="%.1e")
        steps_min = st.number_input("Steps Min", value=512, step=64)
        steps_max = st.number_input("Steps Max", value=2048, step=64)
        batch_min = st.number_input("Batch Size Min", value=64, step=32)
        batch_max = st.number_input("Batch Size Max", value=512, step=32)

    with col2:
        st.subheader("Training Parameters")
        epochs_min = st.number_input("Training Epochs Min", value=3, step=1)
        epochs_max = st.number_input("Training Epochs Max", value=10, step=1)
        gamma_min = st.number_input("Gamma Min", value=0.90, step=0.01, format="%.3f")
        gamma_max = st.number_input("Gamma Max", value=0.999, step=0.001, format="%.3f")
        gae_min = st.number_input("GAE Lambda Min", value=0.90, step=0.01, format="%.2f")
        gae_max = st.number_input("GAE Lambda Max", value=0.99, step=0.01, format="%.2f")

    return {
        "lr": (lr_min, lr_max),
        "steps": (steps_min, steps_max),
        "batch": (batch_min, batch_max),
        "epochs": (epochs_min, epochs_max),
        "gamma": (gamma_min, gamma_max),
        "gae": (gae_min, gae_max),
    }


def run_hyperparameter_optimization(
    stock_names: list,
    train_start_date: datetime,
    train_end_date: datetime,
    env_params: Dict[str, Any],
    param_ranges: Dict[str, Tuple[float, float]],
    trials_number: int,
    optimization_metric: str,
    progress_bar,
    status_text,
    pruning_enabled: bool = True,
    two_phase: bool = True,
    phase1_ratio: float = 0.6,
    iterative_refinement: bool = False,
    improvement_threshold: float = 0.01,
    max_rounds: int = 3,
) -> Tuple[optuna.Study, Optional[TwoPhaseHyperparameterOptimizer]]:
    """
    Run hyperparameter optimization using Optuna.

    Args:
        two_phase: If True, use two-phase optimization
        phase1_ratio: Fraction of trials for Phase 1 (only used if two_phase=True)

    Returns:
        Tuple of (best study, optimizer instance if two_phase else None)
    """
    if two_phase:
        optimizer = TwoPhaseHyperparameterOptimizer(
            stock_names=stock_names,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            env_params=env_params,
            param_ranges=param_ranges,
            optimization_metric=optimization_metric,
            progress_bar=progress_bar,
            status_text=status_text,
        )

        study = optimizer.run_optimization(
            n_trials=trials_number,
            pruning_enabled=pruning_enabled,
            phase1_ratio=phase1_ratio,
            iterative=iterative_refinement,
            max_rounds=max_rounds,
            improvement_threshold=improvement_threshold,
        )

        return study, optimizer

    else:
        # Original single-phase optimization
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner() if pruning_enabled else None,
        )

        def objective(trial: optuna.Trial) -> float:
            try:
                ppo_params = {
                    "learning_rate": trial.suggest_float(
                        "learning_rate", param_ranges["lr"][0], param_ranges["lr"][1], log=True
                    ),
                    "n_steps": trial.suggest_int(
                        "n_steps", int(param_ranges["steps"][0]), int(param_ranges["steps"][1])
                    ),
                    "batch_size": trial.suggest_int(
                        "batch_size", int(param_ranges["batch"][0]), int(param_ranges["batch"][1])
                    ),
                    "n_epochs": trial.suggest_int(
                        "n_epochs", int(param_ranges["epochs"][0]), int(param_ranges["epochs"][1])
                    ),
                    "gamma": trial.suggest_float(
                        "gamma", param_ranges["gamma"][0], param_ranges["gamma"][1]
                    ),
                    "gae_lambda": trial.suggest_float(
                        "gae_lambda", param_ranges["gae"][0], param_ranges["gae"][1]
                    ),
                }

                status_text.text(
                    f"Trial {trial.number + 1}/{trials_number}: "
                    f"Testing parameters {ppo_params}"
                )

                trial_model = UnifiedTradingAgent()
                metrics = trial_model.train(
                    stock_names=stock_names,
                    start_date=train_start_date,
                    end_date=train_end_date,
                    env_params=env_params,
                    ppo_params=ppo_params,
                )

                trial_value = metrics.get(optimization_metric, float("-inf"))
                progress = (trial.number + 1) / trials_number
                progress_bar.progress(progress)

                return trial_value

            except Exception as e:
                logger.error(f"Error in trial {trial.number}: {str(e)}")
                return float("-inf")

        study.optimize(objective, n_trials=trials_number)
        return study, None


def save_best_params(params: Dict[str, Any], value: float, phase: int = None) -> None:
    """Save best parameters to a file."""
    best_params = {
        "params": params,
        "value": value,
        "phase": phase,
        "timestamp": datetime.now().isoformat(),
    }
    with open("best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=2)


def load_best_params() -> Optional[Dict[str, Any]]:
    """Load best parameters from file."""
    try:
        with open("best_hyperparameters.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def display_optimization_results(
    study: optuna.Study, optimizer: Optional[TwoPhaseHyperparameterOptimizer] = None
) -> None:
    """Display optimization results using Streamlit."""

    # Build phase-aware stats early for richer displays
    trials_df: pd.DataFrame
    phase_stats = None
    round_summaries = getattr(optimizer, "round_summaries", []) if optimizer else []

    # Determine which phase produced the best result
    best_phase = None
    if optimizer:
        if getattr(optimizer, "best_phase_overall", None):
            best_phase = optimizer.best_phase_overall
        else:
            if optimizer.phase1_study and optimizer.phase1_study.best_trial:
                if study.best_value == optimizer.phase1_study.best_value:
                    best_phase = 1
            if optimizer.phase2_study and optimizer.phase2_study.best_trial:
                if study.best_value == optimizer.phase2_study.best_value:
                    best_phase = 2

    # Save best parameters
    save_best_params(study.best_params, study.best_value, best_phase)

    # Get trials dataframe
    if optimizer:
        trials_df = optimizer.get_combined_study_data()
        if not trials_df.empty and "Phase" in trials_df.columns:
            phase_stats = (
                trials_df.groupby("Phase")["Value"]
                .agg(["count", "mean", "median", "min", "max"])
                .rename(
                    columns={
                        "count": "Completed", "mean": "Mean", "median": "Median"
                    }
                )
            )
    else:
        trials_df = pd.DataFrame(
            [
                {
                    "Trial": t.number,
                    "Phase": 1,
                    "Value": t.value,
                    **t.params,
                }
                for t in study.trials
                if t.value is not None
            ]
        )

    # Create tabs
    if optimizer:
        tabs = st.tabs(
            [
                "Run Context & Inputs",
                "Best Parameters",
                "Optimization History",
                "Phase Comparison",
                "Parameter Sensitivity",
                "Refined Ranges",
            ]
        )
    else:
        tabs = st.tabs(
            [
                "Run Context & Inputs",
                "Best Parameters",
                "Optimization History",
                "Parameter Importance",
            ]
        )

    # Tab 0: Run Context
    with tabs[0]:
        st.subheader("Fold Inputs & Run Context")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Data Window**")
            st.metric("Train Start", getattr(st.session_state, "train_start_date", "-"))
            st.metric("Train End", getattr(st.session_state, "train_end_date", "-"))
            st.metric("Stocks", len(getattr(st.session_state, "stock_names", [])))

        with col2:
            st.markdown("**Run Configuration**")
            total_trials = optimizer.total_trials if optimizer else len(trials_df)
            st.metric("Total Trials", total_trials)
            if optimizer:
                st.metric("Phase Split", f"{optimizer.phase1_trials_count}/{optimizer.phase2_trials_count}")
                st.metric("Phase 1 Ratio", f"{optimizer.phase1_ratio:.0%}")
            st.metric("Optimization Metric", getattr(st, "session_state", {}).get("optimization_metric", "-"))

        if getattr(st.session_state, "stock_names", None):
            st.markdown("**Stocks Selected**")
            st.write(", ".join(st.session_state.stock_names))

        if optimizer:
            # Show per-phase headline stats
            st.markdown("**Phase-Level Outcome Summary**")
            if phase_stats is not None:
                st.dataframe(
                    phase_stats.style.format({"Mean": "{:.4f}", "Median": "{:.4f}", "min": "{:.4f}", "max": "{:.4f}"}),
                    use_container_width=True,
                )
            else:
                st.info("Phase-level stats will appear once trials finish.")

            if round_summaries:
                st.markdown("**Iterative Round Progress**")
                round_df = pd.DataFrame(round_summaries)
                display_cols = ["round", "phase1_trials", "phase2_trials", "best_value", "improvement"]
                st.dataframe(
                    round_df[display_cols]
                    .rename(
                        columns={
                            "round": "Round",
                            "phase1_trials": "Phase 1 Trials",
                            "phase2_trials": "Phase 2 Trials",
                            "best_value": "Best Value",
                            "improvement": "Δ vs prior",
                        }
                    )
                    .style.format({"Best Value": "{:.4f}", "Δ vs prior": "{:.3f}"}),
                    use_container_width=True,
                )

        # Env inputs snapshot to understand fold inputs
        env_params = getattr(st.session_state, "env_params", {})
        if env_params:
            st.markdown("**Environment Inputs Used in Each Fold**")
            env_df = pd.DataFrame(env_params.items(), columns=["Parameter", "Value"])
            st.dataframe(env_df, use_container_width=True)

    # Tab 1: Best Parameters
    with tabs[1]:
        st.subheader("Best Configuration Found")

        if best_phase:
            st.info(
                f"🏆 Best result from Phase {best_phase} "
                f"({'Exploration' if best_phase == 1 else 'Exploitation'})"
            )

        col1, col2 = st.columns(2)

        with col1:
            for param in ["learning_rate", "gamma", "gae_lambda"]:
                if param in study.best_params:
                    value = study.best_params[param]
                    if param == "learning_rate":
                        st.metric(f"Best {param}", f"{value:.2e}")
                    else:
                        st.metric(f"Best {param}", f"{value:.4f}")

        with col2:
            for param in ["n_steps", "batch_size", "n_epochs"]:
                if param in study.best_params:
                    st.metric(f"Best {param}", f"{int(study.best_params[param])}")

        st.metric("Best Value", f"{study.best_value:.6f}")
        st.session_state.ppo_params = study.best_params

    # Tab 2: Optimization History
    with tabs[2]:
        st.subheader("Trial History")

        history_fig = go.Figure()

        if optimizer and "Phase" in trials_df.columns:
            if "Round" in trials_df.columns:
                for (round_id, phase_id), sub_df in trials_df.groupby(["Round", "Phase"]):
                    color = "blue" if phase_id == 1 else "green"
                    history_fig.add_trace(
                        go.Scatter(
                            x=sub_df["Trial"],
                            y=sub_df["Value"],
                            mode="markers+lines",
                            name=f"Round {round_id} - {'Exploration' if phase_id == 1 else 'Exploitation'}",
                            marker=dict(color=color, size=8),
                            line=dict(color=color, width=1, dash="dot" if phase_id == 1 else "solid"),
                        )
                    )
            else:
                # Plot Phase 1 and Phase 2 separately
                phase1_df = trials_df[trials_df["Phase"] == 1]
                phase2_df = trials_df[trials_df["Phase"] == 2]

                history_fig.add_trace(
                    go.Scatter(
                        x=phase1_df["Trial"],
                        y=phase1_df["Value"],
                        mode="markers+lines",
                        name="Phase 1 (Exploration)",
                        marker=dict(color="blue", size=8),
                        line=dict(color="blue", width=1, dash="dot"),
                    )
                )

                history_fig.add_trace(
                    go.Scatter(
                        x=phase2_df["Trial"],
                        y=phase2_df["Value"],
                        mode="markers+lines",
                        name="Phase 2 (Exploitation)",
                        marker=dict(color="green", size=8),
                        line=dict(color="green", width=1, dash="dot"),
                    )
                )

                # Add phase divider
                if len(phase1_df) > 0 and len(phase2_df) > 0:
                    phase_boundary = phase1_df["Trial"].max() + 0.5
                    history_fig.add_vline(
                        x=phase_boundary,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Phase 2 Start",
                    )
        else:
            history_fig.add_trace(
                go.Scatter(
                    x=trials_df["Trial"],
                    y=trials_df["Value"],
                    mode="lines+markers",
                    name="Trial Value",
                )
            )

        # Add best value line
        history_fig.add_hline(
            y=study.best_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Best: {study.best_value:.4f}",
        )

        history_fig.update_layout(
            title="Optimization History",
            xaxis_title="Trial Number",
            yaxis_title="Metric Value",
            hovermode="x unified",
        )
        st.plotly_chart(history_fig, use_container_width=True)

    # Tab 3: Phase Comparison (only for two-phase)
    if optimizer:
        with tabs[3]:
            st.subheader("Phase Comparison")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Phase 1: Exploration")
                if optimizer.phase1_study and optimizer.phase1_study.best_trial:
                    st.metric("Best Value", f"{optimizer.phase1_study.best_value:.4f}")
                    st.metric("Trials", len(optimizer.phase1_study.trials))

                    completed = len(
                        [
                            t
                            for t in optimizer.phase1_study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE
                        ]
                    )
                    pruned = len(
                        [
                            t
                            for t in optimizer.phase1_study.trials
                            if t.state == optuna.trial.TrialState.PRUNED
                        ]
                    )
                    st.metric("Completed/Pruned", f"{completed}/{pruned}")

            with col2:
                st.markdown("### Phase 2: Exploitation")
                if optimizer.phase2_study and optimizer.phase2_study.best_trial:
                    st.metric("Best Value", f"{optimizer.phase2_study.best_value:.4f}")
                    st.metric("Trials", len(optimizer.phase2_study.trials))

                    completed = len(
                        [
                            t
                            for t in optimizer.phase2_study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE
                        ]
                    )
                    pruned = len(
                        [
                            t
                            for t in optimizer.phase2_study.trials
                            if t.state == optuna.trial.TrialState.PRUNED
                        ]
                    )
                    st.metric("Completed/Pruned", f"{completed}/{pruned}")

            # Box plot comparison
            if "Phase" in trials_df.columns:
                box_fig = go.Figure()
                box_fig.add_trace(
                    go.Box(
                        y=trials_df[trials_df["Phase"] == 1]["Value"],
                        name="Phase 1",
                        boxmean=True,
                    )
                )
                box_fig.add_trace(
                    go.Box(
                        y=trials_df[trials_df["Phase"] == 2]["Value"],
                        name="Phase 2",
                        boxmean=True,
                    )
                )
                box_fig.update_layout(
                    title="Value Distribution by Phase", yaxis_title="Metric Value"
                )
                st.plotly_chart(box_fig, use_container_width=True)

    # Tab 4: Parameter sensitivity / importance
    importance_tab_idx = 4 if optimizer else 3
    with tabs[importance_tab_idx]:
        st.subheader("Parameter Impact Across Folds")

        try:
            importance_dict = optuna.importance.get_param_importances(study)
            importance_df = pd.DataFrame(
                {
                    "Parameter": list(importance_dict.keys()),
                    "Importance": list(importance_dict.values()),
                }
            ).sort_values("Importance", ascending=True)

            importance_fig = go.Figure()
            importance_fig.add_trace(
                go.Bar(
                    x=importance_df["Importance"],
                    y=importance_df["Parameter"],
                    orientation="h",
                    marker_color="steelblue",
                )
            )
            importance_fig.update_layout(
                title="Parameter Importance Analysis",
                xaxis_title="Relative Importance",
                yaxis_title="Parameter",
                height=400,
            )
            st.plotly_chart(importance_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute parameter importance: {str(e)}")

        # Scatter view for a selected parameter to see phase-specific behavior
        if not trials_df.empty:
            available_params = [c for c in trials_df.columns if c not in {"Trial", "Phase", "Value", "State"}]
            if available_params:
                selected_param = st.selectbox(
                    "Inspect parameter vs. metric", available_params, index=0
                )
                scatter_fig = go.Figure()
                if "Phase" in trials_df.columns:
                    for phase_id, phase_name in ((1, "Exploration"), (2, "Exploitation")):
                        phase_subset = trials_df[trials_df["Phase"] == phase_id]
                        scatter_fig.add_trace(
                            go.Scatter(
                                x=phase_subset[selected_param],
                                y=phase_subset["Value"],
                                mode="markers",
                                name=f"Phase {phase_id} ({phase_name})",
                                marker=dict(size=9, opacity=0.75),
                            )
                        )
                else:
                    scatter_fig.add_trace(
                        go.Scatter(
                            x=trials_df[selected_param],
                            y=trials_df["Value"],
                            mode="markers",
                            name=selected_param,
                            marker=dict(size=9, opacity=0.75),
                        )
                    )

                scatter_fig.update_layout(
                    title=f"Effect of {selected_param} on {study.direction.name.title()} metric",
                    xaxis_title=selected_param,
                    yaxis_title="Metric Value",
                    height=450,
                )
                st.plotly_chart(scatter_fig, use_container_width=True)

        # Parallel coordinates for top trials to compare parameter interactions
        if not trials_df.empty and len(trials_df) >= 3:
            top_df = trials_df.nlargest(min(30, len(trials_df)), "Value")
            dimension_cols = [
                col
                for col in top_df.columns
                if col
                not in {
                    "Trial",
                    "Phase",
                    "Value",
                    "State",
                }
            ]
            dimensions = [
                dict(label="Metric", values=top_df["Value"], range=[top_df["Value"].min(), top_df["Value"].max()])
            ]
            for col in dimension_cols:
                try:
                    dimensions.append(dict(label=col, values=top_df[col]))
                except Exception:
                    continue
            if len(dimensions) > 1:
                parallel_fig = go.Figure(
                    data=
                    [
                        go.Parcoords(
                            line=dict(color=top_df["Value"], colorscale="Viridis"),
                            dimensions=dimensions,
                        )
                    ]
                )
                parallel_fig.update_layout(
                    title="Top Trials Parameter Interactions",
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=10),
                )
                st.plotly_chart(parallel_fig, use_container_width=True)

    # Tab 5: Refined Ranges (only for two-phase)
    if optimizer:
        with tabs[5]:
            st.subheader("Refined Parameter Ranges")
            st.markdown(
                """
            These ranges were extracted from the top 20% of Phase 1 trials 
            and used for focused exploration in Phase 2.
            """
            )

            if optimizer.refined_ranges:
                range_data = []
                param_names = {
                    "lr": "Learning Rate",
                    "steps": "N Steps",
                    "batch": "Batch Size",
                    "epochs": "N Epochs",
                    "gamma": "Gamma",
                    "gae": "GAE Lambda",
                }

                for key, (low, high) in optimizer.refined_ranges.items():
                    orig_low, orig_high = optimizer.param_ranges.get(key, (low, high))
                    range_data.append(
                        {
                            "Parameter": param_names.get(key, key),
                            "Original Min": f"{orig_low:.4g}",
                            "Original Max": f"{orig_high:.4g}",
                            "Refined Min": f"{low:.4g}",
                            "Refined Max": f"{high:.4g}",
                            "Narrowed": "✓" if (low > orig_low or high < orig_high) else "",
                        }
                    )

                st.dataframe(pd.DataFrame(range_data), use_container_width=True)
            else:
                st.info("No refined ranges available (using original ranges)")

    # Download results
    trials_df.to_csv("hyperparameter_tuning_results.csv", index=False)
    st.download_button(
        "Download Complete Results CSV",
        trials_df.to_csv(index=False),
        "hyperparameter_tuning_results.csv",
        "text/csv",
    )


def hyperparameter_tuning() -> None:
    """Interface for hyperparameter optimization using Optuna."""

    stock_names = st.session_state.stock_names
    train_start_date = st.session_state.train_start_date
    train_end_date = st.session_state.train_end_date
    env_params = st.session_state.env_params

    st.header("🔧 Hyperparameter Tuning")

    with st.expander("Tuning Configuration", expanded=True):
        # Basic settings
        col1, col2 = st.columns(2)

        with col1:
            trials_number = st.number_input(
                "Total Number of Trials",
                min_value=10,
                value=30,
                step=5,
                help="Total trials across both phases",
            )

            optimization_metric = st.selectbox(
                "Optimization Metric",
                ["sharpe_ratio", "sortino_ratio", "total_return"],
                help="Metric to optimize during hyperparameter search",
            )
            st.session_state.optimization_metric = optimization_metric

        with col2:
            two_phase_enabled = st.checkbox(
                "Enable Two-Phase Optimization",
                value=True,
                help="Phase 1: Broad exploration, Phase 2: Focused exploitation",
            )

            pruning_enabled = st.checkbox("Enable Early Trial Pruning", value=True)

        iterative_refinement = False
        improvement_threshold = 0.0
        max_rounds = 1

        # Two-phase specific settings
        if two_phase_enabled:
            st.markdown("---")
            st.subheader("Two-Phase Settings")

            col1, col2, col3 = st.columns(3)

            with col1:
                phase1_ratio = st.slider(
                    "Phase 1 Ratio",
                    min_value=0.4,
                    max_value=0.8,
                    value=0.6,
                    step=0.1,
                    help="Fraction of trials for exploration phase",
                )

            with col2:
                st.metric("Phase 1 Trials", int(trials_number * phase1_ratio))

            with col3:
                st.metric("Phase 2 Trials", int(trials_number * (1 - phase1_ratio)))

            iterative_refinement = st.checkbox(
                "Iterate until improvement stalls",
                value=True,
                help="Repeat explore→exploit cycles while best metric keeps improving above the threshold.",
            )

            if iterative_refinement:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    max_rounds = st.number_input(
                        "Max refinement rounds", min_value=1, value=3, step=1
                    )
                with col_b:
                    improvement_threshold = st.number_input(
                        "Min improvement to continue",
                        min_value=0.0,
                        value=0.01,
                        step=0.005,
                        format="%.3f",
                        help="Relative improvement required to launch another refinement round.",
                    )
                with col_c:
                    st.metric(
                        "Planned Trials",
                        f"{trials_number * max_rounds} (up to)",
                        help="Product of trials per round and max rounds",
                    )
            else:
                max_rounds = 1
                improvement_threshold = 0.0

            st.info(
                """
            **Two-Phase Optimization:**
            - **Phase 1 (Exploration):** Broadly samples the parameter space to identify promising regions
            - **Phase 2 (Exploitation):** Narrows the search to the top 20% of Phase 1 results with ±10% padding
            """
            )

        st.markdown("---")
        st.subheader("Parameter Ranges")
        param_ranges = create_parameter_ranges()

    if st.button("🚀 Start Hyperparameter Tuning", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            study, optimizer = run_hyperparameter_optimization(
                stock_names=stock_names,
                train_start_date=train_start_date,
                train_end_date=train_end_date,
                env_params=env_params,
                param_ranges=param_ranges,
                trials_number=trials_number,
                optimization_metric=optimization_metric,
                progress_bar=progress_bar,
                status_text=status_text,
                pruning_enabled=pruning_enabled,
                two_phase=two_phase_enabled,
                phase1_ratio=phase1_ratio if two_phase_enabled else 0.6,
                iterative_refinement=iterative_refinement if two_phase_enabled else False,
                improvement_threshold=improvement_threshold,
                max_rounds=max_rounds,
            )

            st.success("✅ Hyperparameter tuning completed!")
            display_optimization_results(study, optimizer)

        except Exception as e:
            st.error(f"❌ Optimization failed: {str(e)}")
            logger.exception("Hyperparameter optimization error")


if __name__ == "__main__":
    # For testing outside Streamlit
    pass
