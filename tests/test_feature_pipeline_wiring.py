from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from core.base_agent import UnifiedTradingAgent
from environment.trading_env import TradingEnv
from data.providers import DataProvider


class DummyProvider(DataProvider):
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def fetch(self, symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:
        return self._frame


def _build_price_frame(symbol: str, rows: int = 5) -> pd.DataFrame:
    base_date = datetime(2024, 1, 1)
    index = [base_date + timedelta(days=i) for i in range(rows)]
    data = {
        f"Open_{symbol}": np.arange(rows, dtype=float) + 1,
        f"High_{symbol}": np.arange(rows, dtype=float) + 2,
        f"Low_{symbol}": np.arange(rows, dtype=float),
        f"Close_{symbol}": np.arange(rows, dtype=float) + 1.5,
        f"Volume_{symbol}": np.arange(rows, dtype=float) + 100,
    }
    return pd.DataFrame(data, index=pd.DatetimeIndex(index)).tz_localize("UTC")


class StubFeatureProcessor:
    def __init__(self, feature_config: Dict[str, Any] | None = None,
                 symbols: List[str] | None = None):
        self.feature_config = feature_config or {}
        self.symbols = symbols or []
        manual = self.feature_config.get('sources', {}).get('manual_pipeline', {})
        self.feature_columns = manual.get('features', ['feat_a', 'feat_b'])
        self._initialized = False

    def initialize(self, data: pd.DataFrame) -> None:
        self._initialized = True

    def compute_features(self, data: pd.DataFrame, current_index: int | None = None) -> pd.DataFrame:
        computed = data.copy()
        for idx, col in enumerate(self.feature_columns):
            computed[col] = float(idx + 1)
        return computed

    def get_observation_size(self) -> int:
        return len(self.feature_columns)

    def get_observation_vector(self, data: pd.DataFrame, current_index: int,
                                positions: Dict[str, float], balance: float) -> np.ndarray:
        row = data.iloc[current_index]
        return np.array([row[col] for col in self.feature_columns], dtype=np.float32)


def test_trading_env_uses_supplied_feature_processor():
    symbol = "TST"
    raw_frame = _build_price_frame(symbol, rows=8)
    provider = DummyProvider(raw_frame)
    processor = StubFeatureProcessor({'use_feature_engineering': True}, [symbol])
    feature_data = processor.compute_features(raw_frame)

    env = TradingEnv(
        stock_names=[symbol],
        start_date=raw_frame.index[0],
        end_date=raw_frame.index[-1],
        observation_days=1,
        burn_in_days=0,
        feature_config={'use_feature_engineering': True},
        feature_processor=processor,
        feature_data=feature_data,
        prefetched_data=raw_frame,
        provider=provider,
    )

    obs, _ = env.reset()

    assert env.feature_processor is processor
    assert env.observation_space.shape[0] == len(processor.feature_columns)
    np.testing.assert_allclose(obs, np.array([1.0, 2.0], dtype=np.float32))


def test_agent_builds_feature_processor_from_pipeline(monkeypatch):
    monkeypatch.setattr(
        "data.feature_engineering.feature_processor.FeatureProcessor",
        StubFeatureProcessor,
    )

    symbol = "AGT"
    raw_frame = _build_price_frame(symbol, rows=40)
    provider = DummyProvider(raw_frame)
    agent = UnifiedTradingAgent()

    feature_config = {'use_feature_engineering': True}
    agent.initialize_env(
        stock_names=[symbol],
        start_date=raw_frame.index[0],
        end_date=raw_frame.index[-1],
        env_params={'history_length': 1},
        feature_config=feature_config,
        feature_pipeline=['alpha', 'beta', 'gamma'],
        training_mode=True,
        provider=provider,
    )

    assert agent.env.feature_processor is not None
    assert agent.env.feature_processor.feature_columns == ['alpha', 'beta', 'gamma']
    assert agent.env.observation_space.shape[0] == 3
