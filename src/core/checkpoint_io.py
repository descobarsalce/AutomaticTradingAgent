"""Checkpoint save/load helpers for PPO agents."""
from __future__ import annotations

import os
from typing import Optional

from stable_baselines3 import PPO

from src.core.experiments import ExperimentRegistry


def save_checkpoint(model: PPO, path: str) -> None:
    if model is None:
        raise ValueError("Model must be initialized before saving")
    if not path.strip():
        raise ValueError("Checkpoint path cannot be empty")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)


def load_checkpoint(path: str, env, manifest_path: Optional[str] = None,
                    enforce_hash: bool = True):
    if not path.strip():
        raise ValueError("Empty path provided for checkpoint")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    manifest = None
    resolved_manifest_path = manifest_path or os.path.join(os.path.dirname(path), "manifest.json")
    if resolved_manifest_path and os.path.exists(resolved_manifest_path):
        manifest = ExperimentRegistry.load_manifest(resolved_manifest_path)

    model = PPO.load(path, env=env)

    return model, manifest
