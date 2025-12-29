import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

from src.core.visualization import TradingVisualizer
from src.metrics.metric_sink import MetricsSink
from src.metrics.metrics_calculator import MetricsCalculator


class MetricStreamingEvalCallback(BaseCallback):
    """Periodic validation episodes that stream metrics to a sink."""

    def __init__(self, eval_env: Any, metrics_sink: MetricsSink,
                 visualizer: TradingVisualizer, eval_freq: int = 1000,
                 seeds: Optional[List[int]] = None,
                 log_dir: str = "metrics/eval") -> None:
        super().__init__()
        self.eval_env = eval_env
        self.metrics_sink = metrics_sink
        self.visualizer = visualizer
        self.eval_freq = eval_freq
        self.seeds = seeds or [0]
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        for seed in self.seeds:
            self._run_validation_episode(seed)
        return True

    def _run_validation_episode(self, seed: int) -> None:
        obs, info = self.eval_env.reset(seed=seed)
        done = False
        info_history: List[Dict[str, Any]] = []

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = terminated or truncated
            info_history.append(info)

        self._log_results(info_history)

    def _log_results(self, info_history: List[Dict[str, Any]]) -> None:
        portfolio_history = self.eval_env.get_portfolio_history()
        returns = MetricsCalculator.calculate_returns(portfolio_history)
        pm = self.eval_env.portfolio_manager

        metrics_payload = {
            'pnl': pm.get_total_value() - pm.initial_balance,
            'max_drawdown': MetricsCalculator.calculate_maximum_drawdown(portfolio_history),
            'turnover': pm.calculate_turnover(),
            'sharpe_ratio': MetricsCalculator.calculate_sharpe_ratio(returns),
            'sortino_ratio': MetricsCalculator.calculate_sortino_ratio(returns),
            'information_ratio': MetricsCalculator.calculate_information_ratio(returns),
            'portfolio': {
                'positions': pm.positions.copy(),
                'balance': pm.current_balance,
                'total_value': pm.get_total_value(),
            }
        }

        gate_history = getattr(self.eval_env, "gate_history", [])
        if gate_history and len(returns) > 0:
            gate_array = np.array(gate_history[:len(returns)], dtype=float)
            metrics_payload['gating'] = {
                'average_gate': float(np.mean(gate_array)),
                'quantile_performance': MetricsCalculator.gate_quantile_performance(
                    gate_array.tolist(), returns),
            }

        if info_history and self.eval_env.data is not None:
            last_date = info_history[-1].get('date', datetime.utcnow())
            filename = f"eval_actions_{last_date.strftime('%Y%m%d_%H%M%S')}"
            snapshot_path = os.path.join(self.log_dir, f"{filename}.html")
            metrics_payload['action_snapshot'] = self.visualizer.snapshot_actions_with_price(
                info_history, self.eval_env.data, snapshot_path)

        self.metrics_sink.emit("validation_episode", metrics_payload)
