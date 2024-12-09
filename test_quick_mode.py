import pandas as pd
import numpy as np
from environment import SimpleTradingEnv
from core import TradingAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Create test data
    logger.info("Creating test data...")
    test_data = pd.DataFrame({
        'Open': [100] * 100,
        'High': [110] * 100,
        'Low': [90] * 100,
        'Close': [105] * 100,
        'Volume': [1000] * 100
    })

    # Create environment
    logger.info("Creating trading environment...")
    env = SimpleTradingEnv(data=test_data, initial_balance=100000)

    # Test quick mode
    logger.info("Initializing quick mode agent...")
    agent = TradingAgent(env, quick_mode=True)
    logger.info(f"Quick mode status: {agent.quick_mode}")
    logger.info(f"Agent parameters: learning_rate={float(str(agent.model.learning_rate))}, "
               f"n_steps={agent.model.n_steps}, batch_size={agent.model.batch_size}")

    # Test basic functionality
    logger.info("Testing basic functionality...")
    obs, info = env.reset()
    action = agent.predict(obs)
    logger.info(f"First action: {action}")

    # Test quick training
    logger.info("Starting quick training...")
    agent.train(total_timesteps=1000)
    logger.info("Quick training completed successfully")

    # Test model prediction after training
    logger.info("Testing model prediction after training...")
    obs, info = env.reset()
    action = agent.predict(obs)
    logger.info(f"Post-training action: {action}")

except Exception as e:
    logger.error(f"Error occurred: {str(e)}", exc_info=True)
    raise
