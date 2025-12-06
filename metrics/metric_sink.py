import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Iterable, Optional


@dataclass
class MetricsSinkConfig:
    jsonl_path: str = "metrics/metrics_stream.jsonl"
    max_drawdown_alert: float = 0.25
    turnover_alert: float = 2.0


class MetricsSink:
    """Simple metrics sink that emits JSONL lines and supports alerting."""

    def __init__(self, config: Optional[MetricsSinkConfig] = None) -> None:
        self.config = config or MetricsSinkConfig()
        directory = os.path.dirname(self.config.jsonl_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        enriched = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "payload": payload,
            "alerts": self._collect_alerts(payload),
        }
        with open(self.config.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(enriched, default=str) + "\n")

    def _collect_alerts(self, payload: Dict[str, Any]) -> Iterable[str]:
        alerts = []
        max_dd = payload.get("max_drawdown")
        turnover = payload.get("turnover")
        if max_dd is not None and max_dd >= self.config.max_drawdown_alert:
            alerts.append(f"max_drawdown_threshold:{max_dd}")
        if turnover is not None and turnover >= self.config.turnover_alert:
            alerts.append(f"turnover_threshold:{turnover}")
        return alerts
