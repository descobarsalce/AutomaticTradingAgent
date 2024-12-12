
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from .base import BaseEnvironment
from gymnasium import spaces

class MarketEnvironment(BaseEnvironment):
    def __init__(self, data: Dict[str, pd.DataFrame], weights: Dict[str, float]):
        super().__init__()
        self.data = data
        self.weights = weights
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(len(weights),), dtype=np.float32
        )
        obs_dim = len(weights) * 5  # 5 features per asset
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
