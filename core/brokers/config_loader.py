"""Helpers to load broker configuration for IBKR/Lean sessions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from core.brokers.ibkr_adapter import IBKRConfig


def load_ibkr_config(path: str | os.PathLike[str]) -> IBKRConfig:
    """Load IBKR configuration from a YAML file.

    Expected keys:
    - host, port, client_id, account, readonly, connect_timeout, auto_disconnect, log_level
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"IBKR config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    return IBKRConfig(
        host=raw.get("host", "127.0.0.1"),
        port=int(raw.get("port", 4002)),
        client_id=int(raw.get("client_id", 1)),
        account=raw.get("account"),
        readonly=bool(raw.get("readonly", False)),
        connect_timeout=int(raw.get("connect_timeout", 10)),
        auto_disconnect=bool(raw.get("auto_disconnect", True)),
        log_level=int(raw.get("log_level", 20)),
    )
