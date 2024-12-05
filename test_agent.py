import gymnasium as gym
import numpy as np
from agent import TradingAgent
import pandas as pd

# Create a simple custom environment for testing
class SimpleTradingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(3)  # -1, 0, 1 for sell, hold, buy
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )
        self.current_step = 0
        self.max_steps = 100
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        observation = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return observation, {}
        
    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Map discrete action to -1, 0, 1
        action_map = {0: -1, 1: 0, 2: 1}
        mapped_action = action_map[action]
        
        reward = float(mapped_action)  # Simple reward based on action
        next_state = np.array(np.random.randn(4), dtype=np.float32)
        
        return next_state, reward, done, False, {}

# Create test environment
env = SimpleTradingEnv()

# Test custom PPO parameters
custom_params = {
    'learning_rate': 0.0003,
    'n_steps': 2048,
    'batch_size': 128,
    'n_epochs': 5,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2
}

def run_tests():
    print("Starting agent tests...")
    
    # Test 1: Initialize agent with custom parameters
    print("\nTest 1: Initializing agent with custom parameters...")
    agent = TradingAgent(env, ppo_params=custom_params)
    print("Agent initialized successfully")
    
    # Test 2: Training
    print("\nTest 2: Training agent...")
    agent.train(total_timesteps=5000)  # Small number for testing
    print("Training completed")
    
    # Test 3: State tracking
    print("\nTest 3: Testing state tracking...")
    initial_portfolio = 10000.0
    positions = {'AAPL': 0.6, 'GOOGL': 0.4}
    
    agent.update_state(
        portfolio_value=initial_portfolio,
        positions=positions
    )
    
    # Update state multiple times to generate history
    for i in range(5):
        new_portfolio = initial_portfolio * (1 + np.random.normal(0, 0.01))
        agent.update_state(
            portfolio_value=new_portfolio,
            positions={'AAPL': 0.6 + np.random.normal(0, 0.01),
                      'GOOGL': 0.4 + np.random.normal(0, 0.01)}
        )
    
    # Test 4: Metrics calculation
    print("\nTest 4: Testing metrics calculation...")
    agent.calculate_metrics()
    
    print("\nEvaluation metrics:")
    print(f"Sharpe Ratio: {agent.evaluation_metrics['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {agent.evaluation_metrics['max_drawdown']:.4f}")
    print(f"Number of returns tracked: {len(agent.evaluation_metrics['returns'])}")
    
    # Test 5: Save and load
    print("\nTest 5: Testing save and load functionality...")
    save_path = "test_agent_model"
    agent.save(save_path)
    
    # Load the saved model
    new_agent = TradingAgent(env)
    new_agent.load(save_path)
    
    print("Tests completed successfully!")

if __name__ == "__main__":
    run_tests()
