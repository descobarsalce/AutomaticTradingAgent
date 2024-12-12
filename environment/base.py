
from typing import Dict, Any, Tuple, List
import gymnasium as gym
import numpy as np

class BaseEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        self.initialized = False
        
    def initialize(self, config: Dict[str, Any]) -> None:
        if self.initialized:
            return
        self.initialized = True
        self._validate_config(config)
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required_fields = ['observation_space', 'action_space']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
