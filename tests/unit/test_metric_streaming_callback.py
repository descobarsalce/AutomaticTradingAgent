from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
import types
import pytest

pytest.importorskip("numpy")

sys.path.append(str(Path(__file__).resolve().parents[1]))


class _DummyBaseCallback:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.n_calls = 0
        self.model = None


fake_callbacks_module = types.SimpleNamespace(BaseCallback=_DummyBaseCallback)
sys.modules.setdefault("stable_baselines3", types.SimpleNamespace(common=types.SimpleNamespace(callbacks=fake_callbacks_module)))
sys.modules.setdefault("stable_baselines3.common", types.SimpleNamespace(callbacks=fake_callbacks_module))
sys.modules.setdefault("stable_baselines3.common.callbacks", fake_callbacks_module)


class _DummyMetricsCalculator:
    @staticmethod
    def calculate_returns(portfolio_history: List[float]):
        return portfolio_history

    @staticmethod
    def calculate_maximum_drawdown(portfolio_history: List[float]) -> float:
        return 0.0

    @staticmethod
    def calculate_sharpe_ratio(returns: List[float]) -> float:
        return 0.0

    @staticmethod
    def calculate_sortino_ratio(returns: List[float]) -> float:
        return 0.0

    @staticmethod
    def calculate_information_ratio(returns: List[float]) -> float:
        return 0.0


sys.modules.setdefault("src.metrics.metrics_calculator", types.SimpleNamespace(MetricsCalculator=_DummyMetricsCalculator))


class _DummyTradingVisualizer:
    def snapshot_actions_with_price(self, info_history: Any, data: Any, path: str) -> str:
        return path


sys.modules.setdefault("src.core.visualization", types.SimpleNamespace(TradingVisualizer=_DummyTradingVisualizer))

from src.core.callbacks.metric_streaming import MetricStreamingEvalCallback


class DummyModel:
    def predict(self, observation: Any, deterministic: bool = True) -> Tuple[int, None]:
        return 0, None


class DummyPortfolioManager:
    def __init__(self) -> None:
        self.initial_balance = 100.0
        self.current_balance = 90.0
        self.positions: Dict[str, float] = {"AAPL": 1.0}

    def get_total_value(self) -> float:
        return 110.0

    def calculate_turnover(self) -> float:
        return 0.2


class DummyEnv:
    def __init__(self, steps: int = 1) -> None:
        self.steps = steps
        self.data = [1, 2, 3]
        self._reset_calls: List[Optional[int]] = []
        self.portfolio_manager = DummyPortfolioManager()

    def reset(self, seed: Optional[int] = None) -> Tuple[int, Dict[str, Any]]:
        self._reset_calls.append(seed)
        self._step_count = 0
        return 0, {"date": datetime(2024, 1, 1)}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        done = self._step_count >= self.steps
        return 0, 1.0, done, False, {"date": datetime(2024, 1, 1, 0, 0, self._step_count)}

    def get_portfolio_history(self) -> List[float]:
        return [100.0, 110.0]


class DummyVisualizer:
    def __init__(self) -> None:
        self.snapshot_calls: List[Tuple[List[Dict[str, Any]], Any, str]] = []

    def snapshot_actions_with_price(self, info_history: List[Dict[str, Any]], data: Any, path: str) -> str:
        self.snapshot_calls.append((info_history, data, path))
        return path


class RecordingMetricsSink:
    def __init__(self) -> None:
        self.events: List[Tuple[str, Dict[str, Any]]] = []

    def emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.events.append((event_type, payload))


def test_eval_runs_at_configured_frequency(tmp_path) -> None:
    env = DummyEnv()
    sink = RecordingMetricsSink()
    callback = MetricStreamingEvalCallback(env, sink, DummyVisualizer(), eval_freq=3, log_dir=tmp_path)
    callback.model = DummyModel()

    callback.n_calls = 1
    callback._on_step()
    assert sink.events == []

    callback.n_calls = 3
    callback._on_step()
    assert sink.events, "Expected metrics to be emitted when frequency threshold reached"


def test_seeds_are_used_for_each_validation_run(tmp_path) -> None:
    env = DummyEnv()
    sink = RecordingMetricsSink()
    seeds = [11, 22]
    callback = MetricStreamingEvalCallback(env, sink, DummyVisualizer(), eval_freq=1, seeds=seeds, log_dir=tmp_path)
    callback.model = DummyModel()

    callback.n_calls = 1
    callback._on_step()

    assert env._reset_calls == seeds


def test_metrics_emitted_include_portfolio_and_snapshot(tmp_path) -> None:
    env = DummyEnv()
    sink = RecordingMetricsSink()
    visualizer = DummyVisualizer()
    callback = MetricStreamingEvalCallback(env, sink, visualizer, eval_freq=1, log_dir=tmp_path)
    callback.model = DummyModel()

    callback.n_calls = 1
    callback._on_step()

    assert sink.events[0][0] == "validation_episode"
    payload = sink.events[0][1]

    assert payload["pnl"] == env.portfolio_manager.get_total_value() - env.portfolio_manager.initial_balance
    assert payload["turnover"] == env.portfolio_manager.calculate_turnover()
    assert payload["portfolio"]["positions"] == env.portfolio_manager.positions
    assert "action_snapshot" in payload
    assert visualizer.snapshot_calls, "Snapshot should be generated when info history and data are available"
