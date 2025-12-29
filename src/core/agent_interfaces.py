"""Typed interfaces and configuration objects for trading agents."""
from __future__ import annotations

from dataclasses import src.dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol

from stable_baselines3.common.callbacks import BaseCallback

from src.data.providers import DataProvider


class FeaturePipeline(Protocol):
    """Protocol for feature pipelines injected into the environment."""

    def initialize(self, data):  # pragma: no cover - interface only
        ...

    def compute_features(self, data):  # pragma: no cover - interface only
        ...


@dataclass
class EnvConfig:
    """Configuration for building a TradingEnv instance."""

    stock_names: List[str]
    start_date: datetime
    end_date: datetime
    params: Dict[str, Any]
    feature_config: Optional[Dict[str, Any]] = None
    feature_pipeline: Optional[Any] = None
    training_mode: bool = True
    provider: Optional[DataProvider] = None


@dataclass
class TrainingSchedule:
    episode_length: int
    warmup_steps: int
    total_timesteps: int


@dataclass
class EvaluationHooks:
    """Container for dependency-injected evaluation callbacks."""

    callbacks: List[BaseCallback] = field(default_factory=list)
    builder: Optional[Callable[[Any], BaseCallback]] = None

    def build(self, env) -> List[BaseCallback]:
        hooks: List[BaseCallback] = list(self.callbacks)
        if self.builder is not None:
            hooks.append(self.builder(env))
        return hooks


@dataclass
class AgentRuntimeConfig:
    """Aggregate runtime configuration for the unified trading agent."""

    env: EnvConfig
    ppo_params: Dict[str, Any]
    evaluation_hooks: EvaluationHooks = field(default_factory=EvaluationHooks)
    checkpoint_dir: str = "artifacts/checkpoints"
    manifest_filename: str = "manifest.json"
    deterministic_eval: bool = True
