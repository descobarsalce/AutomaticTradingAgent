import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import itertools
import logging
from agent import TradingAgent
from gymnasium import Env

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class HyperparameterOptimizer:
    def __init__(self, env: Env, param_grid: Optional[Dict[str, List[Any]]] = None):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            env: The trading environment
            param_grid: Dictionary of parameters and their possible values for grid search
        """
        self.env = env
        self.best_params = None
        self.best_reward = float('-inf')
        self.results = []
        
        # Use default parameter grid if none provided
        if param_grid is None:
            self.param_grid = {
                'learning_rate': [1e-5, 3e-4, 1e-3],
                'n_steps': [512, 1024, 2048],
                'batch_size': [32, 64, 128],
                'clip_range': [0.1, 0.2, 0.3],
                'ent_coef': [0.0, 0.01, 0.05],
            }
        else:
            self.param_grid = param_grid

    def generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all possible combinations of parameters for grid search."""
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        return [dict(zip(param_names, combo)) for combo in combinations]

    def evaluate_params(self, agent: TradingAgent, n_episodes: int = 5) -> float:
        """
        Evaluate a set of parameters by running multiple episodes and averaging the rewards.
        
        Args:
            agent: Trained trading agent to evaluate
            n_episodes: Number of episodes to run for evaluation
            
        Returns:
            Average reward across episodes
        """
        total_reward = 0.0
        
        for episode in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action = agent.predict(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            total_reward += episode_reward
            
        return total_reward / n_episodes

    def optimize(self, total_timesteps: int = 10000, n_eval_episodes: int = 5) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Perform grid search to find optimal hyperparameters.
        
        Args:
            total_timesteps: Number of timesteps to train each agent
            n_eval_episodes: Number of episodes to evaluate each parameter combination
            
        Returns:
            Tuple of (best parameters, list of all results)
        """
        param_combinations = self.generate_param_combinations()
        total_combinations = len(param_combinations)
        
        logger.info(f"Starting grid search with {total_combinations} parameter combinations")
        
        for idx, params in enumerate(param_combinations, 1):
            logger.info(f"Testing combination {idx}/{total_combinations}")
            logger.info(f"Parameters: {params}")
            
            try:
                # Initialize agent with current parameters
                agent = TradingAgent(self.env, ppo_params=params)
                
                # Train agent
                agent.train(total_timesteps)
                
                # Evaluate agent
                avg_reward = self.evaluate_params(agent, n_eval_episodes)
                
                # Store results
                result = {
                    'params': params,
                    'avg_reward': avg_reward,
                    'sharpe_ratio': agent.get_metrics()['sharpe_ratio']
                }
                self.results.append(result)
                
                # Update best parameters if needed
                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    self.best_params = params
                    logger.info(f"New best parameters found! Average reward: {avg_reward}")
                    logger.info(f"Best parameters: {params}")
                
            except Exception as e:
                logger.error(f"Error evaluating parameters: {str(e)}")
                continue
        
        # Sort results by average reward
        self.results.sort(key=lambda x: x['avg_reward'], reverse=True)
        
        return self.best_params, self.results

    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the optimization results.
        
        Returns:
            Dictionary containing optimization summary
        """
        if not self.results:
            return {"status": "No optimization results available"}
        
        return {
            "best_params": self.best_params,
            "best_reward": self.best_reward,
            "total_combinations_tested": len(self.results),
            "top_5_results": self.results[:5]
        }
