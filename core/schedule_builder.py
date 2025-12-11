"""Training schedule utilities for the trading agent."""
from __future__ import annotations

from typing import Any, Dict

from core.agent_interfaces import TrainingSchedule


DEFAULT_SCHEDULE_CONFIG = {
    "episode_length": 256,
    "epochs": 3,
    "warmup_fraction": 0.1,
}


def build_training_schedule(data_points: int,
                            schedule_config: Dict[str, Any] | None = None) -> TrainingSchedule:
    """Derive episode and rollout counts from dataset characteristics."""
    config = DEFAULT_SCHEDULE_CONFIG.copy()
    if schedule_config:
        config.update({k: v for k, v in schedule_config.items() if v is not None})

    episode_length = max(1, min(config["episode_length"], data_points))
    warmup_steps = int(data_points * config["warmup_fraction"])
    total_timesteps = warmup_steps + (episode_length * config["epochs"])

    return TrainingSchedule(
        episode_length=episode_length,
        warmup_steps=warmup_steps,
        total_timesteps=max(total_timesteps, data_points),
    )
