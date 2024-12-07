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
    def __init__(self, env: Env, param_grid: Optional[Dict[str, List[Any]]] = None, fast_mode: bool = False):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            env: The trading environment
            param_grid: Dictionary of parameters and their possible values for grid search
            fast_mode: If True, use a reduced parameter grid for faster optimization
        """
        self.env = env
        self.best_params = None
        self.best_reward = float('-inf')
        self.results = []
        self.early_stop_threshold = 0.8  # Early stopping if 80% of max possible reward reached
        self.patience = 3  # Number of trials without improvement before early stopping
        self.min_improvement = 0.05  # Minimum improvement required (5%)
        
        # Use default parameter grid if none provided
        if param_grid is None:
            if fast_mode:
                # Reduced parameter grid for faster optimization
                self.param_grid = {
                    'learning_rate': [3e-4],  # Most common optimal value
                    'n_steps': [1024],        # Standard value
                    'batch_size': [64],       # Standard value
                    'clip_range': [0.2],      # Standard value
                }
            else:
                # Focused parameter grid with most impactful parameters
                self.param_grid = {
                    'learning_rate': [1e-4, 3e-4],  # Focused around common optimal values
                    'n_steps': [512, 1024],         # Most impactful for trading
                    'batch_size': [64, 128],        # Common optimal range
                    'clip_range': [0.1, 0.2],       # Standard range
                }
        else:
            self.param_grid = param_grid

    def generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all possible combinations of parameters for grid search."""
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        return [dict(zip(param_names, combo)) for combo in combinations]

    def evaluate_params(self, agent: TradingAgent, n_episodes: int = 5, early_stop: bool = True) -> Tuple[float, bool]:
        """
        Evaluate a set of parameters by running multiple episodes and averaging the rewards.
        
        Args:
            agent: Trained trading agent to evaluate
            n_episodes: Number of episodes to run for evaluation
            early_stop: Whether to enable early stopping during evaluation
            
        Returns:
            Tuple of (average reward, early_stop_flag)
        """
        total_reward = 0.0
        rewards = []
        early_stop_flag = False
        min_episodes = max(2, n_episodes // 3)  # Minimum episodes before early stopping
        
        for episode in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action = agent.predict(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            rewards.append(episode_reward)
            total_reward += episode_reward
            
            # Early stopping logic
            if early_stop and episode >= min_episodes:
                current_avg = total_reward / (episode + 1)
                reward_std = np.std(rewards) if len(rewards) > 1 else 0
                
                # Check if performance is significantly poor
                if current_avg < self.best_reward * 0.7:  # 30% worse than best
                    early_stop_flag = True
                    break
                
                # Check if performance is stable enough
                if reward_std / (abs(current_avg) + 1e-8) < 0.1:  # Low variance
                    break
        
        return total_reward / len(rewards), early_stop_flag

    def optimize(self, total_timesteps: int = 10000, n_eval_episodes: int = 5, 
              fast_mode: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Perform grid search to find optimal hyperparameters with early stopping.
        
        Args:
            total_timesteps: Number of timesteps to train each agent
            n_eval_episodes: Number of episodes to evaluate each parameter combination
            fast_mode: If True, use faster evaluation with early stopping
            
        Returns:
            Tuple of (best parameters, list of all results)
        """
        param_combinations = self.generate_param_combinations()
        total_combinations = len(param_combinations)
        trials_without_improvement = 0
        previous_best = float('-inf')
        
        logger.info(f"Starting grid search with {total_combinations} parameter combinations")
        
        for idx, params in enumerate(param_combinations, 1):
            logger.info(f"Testing combination {idx}/{total_combinations}")
            logger.info(f"Parameters: {params}")
            
            try:
                # Initialize agent with current parameters
                agent = TradingAgent(self.env, ppo_params=params)
                
                # Faster training for initial evaluation
                if fast_mode:
                    initial_train_steps = total_timesteps // 2
                    agent.train(initial_train_steps)
                    avg_reward, should_stop = self.evaluate_params(
                        agent, 
                        n_episodes=max(2, n_eval_episodes // 2),
                        early_stop=True
                    )
                    
                    # If initial performance is promising, complete training
                    if not should_stop and avg_reward >= self.best_reward * 0.8:
                        agent.train(total_timesteps - initial_train_steps)
                        avg_reward, _ = self.evaluate_params(
                            agent,
                            n_episodes=n_eval_episodes,
                            early_stop=False
                        )
                else:
                    # Full training
                    agent.train(total_timesteps)
                    avg_reward, should_stop = self.evaluate_params(
                        agent,
                        n_episodes=n_eval_episodes,
                        early_stop=True
                    )
                
                if should_stop:
                    logger.info("Early stopping for this parameter combination due to poor performance")
                    continue
                
                # Store results
                result = {
                    'params': params,
                    'avg_reward': avg_reward,
                    'sharpe_ratio': agent.get_metrics()['sharpe_ratio']
                }
                self.results.append(result)
                
                # Update best parameters if needed
                if avg_reward > self.best_reward:
                    improvement = (avg_reward - self.best_reward) / (abs(self.best_reward) + 1e-8)
                    if improvement > self.min_improvement:
                        self.best_reward = avg_reward
                        self.best_params = params
                        previous_best = self.best_reward
                        trials_without_improvement = 0
                        logger.info(f"New best parameters found! Average reward: {avg_reward}")
                        logger.info(f"Best parameters: {params}")
                else:
                    trials_without_improvement += 1
                
                # Early stopping if no improvement for several trials
                if trials_without_improvement >= self.patience:
                    logger.info("Early stopping: No significant improvement for several trials")
                    break
                
                # Early stopping if we reached a good enough reward
                theoretical_max = self.env.reward_range[1] if hasattr(self.env, 'reward_range') else float('inf')
                if self.best_reward >= theoretical_max * self.early_stop_threshold:
                    logger.info("Early stopping: Reached reward threshold")
                    break
                
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
            return {
                "status": "No optimization results available",
                "best_params": {},
                "best_reward": 0.0,
                "total_combinations_tested": 0,
                "top_5_results": []
            }
        
        return {
            "status": "Optimization completed",
            "best_params": self.best_params if self.best_params is not None else {},
            "best_reward": self.best_reward,
            "total_combinations_tested": len(self.results),
            "top_5_results": self.results[:5]
        }
