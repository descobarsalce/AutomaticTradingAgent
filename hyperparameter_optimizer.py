import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import itertools
import logging
import time
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
        self.best_params = {}  # Initialize as empty dict instead of None
        self.best_reward = float('-inf')
        self.results = []
        self.early_stop_threshold = 0.8  # Early stopping if 80% of max possible reward reached
        self.patience = 3  # Number of trials without improvement before early stopping
        self.min_improvement = 0.05  # Minimum improvement required (5%)
        self.optimization_status = "Not started"  # Track optimization status
        
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
        Implements progressive evaluation with early stopping for efficiency.
        
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
        
        # Track performance metrics for early stopping
        best_episode_reward = float('-inf')
        consecutive_poor_episodes = 0
        reward_threshold = self.best_reward * 0.8 if self.best_reward > float('-inf') else None
        
        for episode in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_trades = 0
            episode_rejected_trades = 0
            
            while not done:
                action = agent.predict(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
                # Track trade statistics
                if 'trade_status' in info:
                    episode_trades += 1
                    if 'rejected' in info['trade_status']:
                        episode_rejected_trades += 1
            
            rewards.append(episode_reward)
            total_reward += episode_reward
            
            # Update best episode reward
            best_episode_reward = max(best_episode_reward, episode_reward)
            
            # Early stopping logic with enhanced criteria
            if early_stop and episode >= min_episodes:
                current_avg = total_reward / (episode + 1)
                reward_std = np.std(rewards) if len(rewards) > 1 else 0
                
                # Check multiple stopping conditions
                should_stop = False
                
                # 1. Performance significantly worse than best known
                if reward_threshold and current_avg < reward_threshold:
                    consecutive_poor_episodes += 1
                    if consecutive_poor_episodes >= 2:  # Two consecutive poor episodes
                        logger.info(f"Stopping early: Performance below threshold for {consecutive_poor_episodes} episodes")
                        early_stop_flag = True
                        should_stop = True
                else:
                    consecutive_poor_episodes = 0
                
                # 2. Too many rejected trades
                if episode_trades > 0 and episode_rejected_trades / episode_trades > 0.8:
                    logger.info("Stopping early: Too many rejected trades")
                    early_stop_flag = True
                    should_stop = True
                
                # 3. Stable performance with low variance
                if reward_std / (abs(current_avg) + 1e-8) < 0.05 and episode >= min_episodes * 2:
                    logger.info("Stopping early: Performance stabilized")
                    should_stop = True
                
                # 4. No improvement trend
                if episode > 2 and best_episode_reward < max(rewards[:-1]):
                    consecutive_poor_episodes += 1
                    if consecutive_poor_episodes >= 3:
                        logger.info("Stopping early: No improvement trend")
                        early_stop_flag = True
                        should_stop = True
                
                if should_stop:
                    break
        
        return total_reward / len(rewards), early_stop_flag

    def _quick_evaluate(self, agent: TradingAgent, n_episodes: int = 2) -> float:
        """Quickly evaluates an agent with minimal episodes."""
        total_reward = 0
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = agent.predict(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            total_reward += episode_reward
        return total_reward / n_episodes

    def optimize(self, total_timesteps: int = 10000, n_eval_episodes: int = 5,
              fast_mode: bool = False, progress_bar=None, status_placeholder=None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Perform grid search to find optimal hyperparameters with early stopping.
        
        Args:
            total_timesteps: Number of timesteps to train each agent
            n_eval_episodes: Number of episodes to evaluate each parameter combination
            fast_mode: If True, use faster evaluation with early stopping
            
        Returns:
            Tuple of (best parameters, list of all results)
        """
        self.optimization_status = "In progress"
        # Initialize with default parameters in case optimization fails
        self.best_params = {
            'learning_rate': 3e-4,
            'n_steps': 1024,
            'batch_size': 64,
            'clip_range': 0.2,
        }
        param_combinations = self.generate_param_combinations()
        total_combinations = len(param_combinations)
        trials_without_improvement = 0
        previous_best = float('-inf')
        
        logger.info(f"Starting grid search with {total_combinations} parameter combinations")
        start_time = time.time()
        
        if progress_bar is not None:
            progress_bar.progress(0.0)
        if status_placeholder is not None:
            status_placeholder.text("Starting hyperparameter optimization...")
        
        for idx, params in enumerate(param_combinations, 1):
            logger.info(f"Testing combination {idx}/{total_combinations}")
            logger.info(f"Parameters: {params}")
            
            # Update progress
            if progress_bar is not None:
                progress = (idx - 1) / total_combinations
                progress_bar.progress(progress)
            
            # Update status with time estimation
            if status_placeholder is not None:
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time * (total_combinations / idx)
                remaining_time = estimated_total_time - elapsed_time
                
                # Format time string
                if remaining_time < 60:
                    time_str = f"{remaining_time:.0f} seconds"
                elif remaining_time < 3600:
                    time_str = f"{remaining_time/60:.1f} minutes"
                else:
                    time_str = f"{remaining_time/3600:.1f} hours"
                
                status_placeholder.text(
                    f"Optimization Progress: {progress*100:.1f}%\n"
                    f"Testing combination {idx}/{total_combinations}\n"
                    f"Estimated time remaining: {time_str}"
                )
            
            try:
                # Initialize agent with current parameters
                agent = TradingAgent(self.env, ppo_params=params)
                
                # Streamlined fast evaluation
                if fast_mode:
                    # Ultra-quick single evaluation
                    train_steps = min(total_timesteps // 16, 250)  # Minimal training
                    agent.train(train_steps)
                    
                    # Single evaluation pass
                    reward = self._quick_evaluate(agent, n_episodes=2)
                    
                    # Simple threshold check
                    if self.best_reward > float('-inf') and reward < self.best_reward * 0.4:
                        continue
                    
                    # Record result
                    avg_reward = reward
                else:
                    # Standard full training with early stopping
                    agent.train(total_timesteps)
                    avg_reward, should_stop = self.evaluate_params(
                        agent,
                        n_episodes=n_eval_episodes,
                        early_stop=True
                    )
                
                
                
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
                self.optimization_status = f"Failed: {str(e)}"
                continue
        
        # Sort results by average reward
        self.results.sort(key=lambda x: x['avg_reward'], reverse=True)
        
        return self.best_params, self.results

    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the optimization results with detailed status information.
        
        Returns:
            Dictionary containing optimization summary
        """
        if not self.results:
            return {
                "status": self.optimization_status,
                "message": "No optimization results available. The process may have failed or been interrupted.",
                "best_params": self.best_params,  # Return default params if optimization failed
                "best_reward": self.best_reward if self.best_reward != float('-inf') else 0.0,
                "total_combinations_tested": 0,
                "top_5_results": [],
                "success": False
            }
        
        return {
            "status": "Optimization completed successfully",
            "message": f"Successfully tested {len(self.results)} parameter combinations",
            "best_params": self.best_params,
            "best_reward": self.best_reward,
            "total_combinations_tested": len(self.results),
            "top_5_results": self.results[:5],
            "success": True
        }