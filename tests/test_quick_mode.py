
import pandas as pd
import numpy as np
from environment import SimpleTradingEnv
from core import TradingAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_quick_mode():
    """Simplified quick mode test"""
    # Create minimal test data
    test_data = pd.DataFrame({
        'Open': [100] * 50,
        'High': [110] * 50,
        'Low': [90] * 50,
        'Close': [105] * 50,
        'Volume': [1000] * 50
    })

    # Initialize environment and agent
    env = SimpleTradingEnv(data=test_data, initial_balance=10000)
    agent = TradingAgent(env, quick_mode=True)

    # Basic functionality test
    obs, _ = env.reset()
    action = agent.predict(obs)
    logger.info(f"Initial action: {action}")

    # Quick training test
    agent.train(total_timesteps=500)
    
    # Verify post-training behavior
    obs, _ = env.reset()
    action = agent.predict(obs)
    logger.info(f"Post-training action: {action}")

if __name__ == "__main__":
    test_quick_mode()
