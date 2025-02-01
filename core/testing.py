"""
Testing module for evaluating trained models and tracking performance metrics.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import json

try:
    import numpy as np
    import pandas as pd
    from stable_baselines3 import PPO
    import yfinance as yf
except ImportError as e:
    logging.error(f"Failed to import required packages: {str(e)}")
    raise

from config import TESTING_CONFIG, PATHS

class ModelTester:
    def __init__(self, env, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model tester.
        
        Args:
            env: The gymnasium environment for testing
            config: Optional configuration override
        """
        self.env = env
        self.config = {**TESTING_CONFIG, **(config or {})}
        self.setup_logging()
        self.results = []
        
    def setup_logging(self):
        """Configure logging for testing process."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{PATHS['logs']}/testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def calculate_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate various performance metrics from returns.
        
        Args:
            returns: Array of period returns
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        # Avoid division by zero and handle empty returns
        if len(returns) == 0:
            return {metric: 0.0 for metric in self.config["metrics"]}
            
        # Basic metrics
        metrics["total_return"] = np.prod(1 + returns) - 1
        metrics["avg_return"] = np.mean(returns)
        metrics["std_dev"] = np.std(returns)
        
        # Sharpe Ratio
        if "sharpe_ratio" in self.config["metrics"]:
            excess_returns = returns - self.config["risk_free_rate"] / 252  # Daily risk-free rate
            if metrics["std_dev"] != 0:
                metrics["sharpe_ratio"] = np.sqrt(252) * np.mean(excess_returns) / metrics["std_dev"]
            else:
                metrics["sharpe_ratio"] = 0.0
        
        # Sortino Ratio
        if "sortino_ratio" in self.config["metrics"]:
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
            if downside_std != 0:
                metrics["sortino_ratio"] = np.sqrt(252) * metrics["avg_return"] / downside_std
            else:
                metrics["sortino_ratio"] = 0.0
        
        # Maximum Drawdown
        if "max_drawdown" in self.config["metrics"]:
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns / running_max - 1
            metrics["max_drawdown"] = np.min(drawdowns)
        
        # Win Rate
        if "win_rate" in self.config["metrics"]:
            metrics["win_rate"] = np.mean(returns > 0)
        
        # Profit Factor
        if "profit_factor" in self.config["metrics"]:
            gains = returns[returns > 0].sum()
            losses = abs(returns[returns < 0].sum())
            metrics["profit_factor"] = gains / losses if losses != 0 else float('inf')
        
        # Calmar Ratio
        if "calmar_ratio" in self.config["metrics"]:
            if metrics["max_drawdown"] != 0:
                metrics["calmar_ratio"] = -metrics["total_return"] / metrics["max_drawdown"]
            else:
                metrics["calmar_ratio"] = 0.0
                
        return metrics

    def evaluate_model(self, model_path: str, n_episodes: int = 1) -> Dict[str, Any]:
        """
        Evaluate a trained model over multiple episodes.
        
        Args:
            model_path: Path to the saved model
            n_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            model = PPO.load(model_path)
            self.logger.info(f"Loaded model from {model_path}")
            
            all_returns = []
            episode_rewards = []
            
            for episode in range(n_episodes):
                self.logger.info(f"Starting evaluation episode {episode + 1}/{n_episodes}")
                
                obs = self.env.reset()
                done = False
                episode_reward = 0
                returns = []
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.env.step(action)
                    
                    episode_reward += reward
                    if "return" in info:
                        returns.append(info["return"])
                
                episode_rewards.append(episode_reward)
                all_returns.extend(returns)
                
                self.logger.info(f"Episode {episode + 1} completed with reward: {episode_reward}")
            
            # Calculate metrics
            returns_array = np.array(all_returns)
            metrics = self.calculate_metrics(returns_array)
            
            # Prepare results
            results = {
                "model_path": model_path,
                "n_episodes": n_episodes,
                "avg_episode_reward": np.mean(episode_rewards),
                "std_episode_reward": np.std(episode_rewards),
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save results
            results_path = os.path.join(PATHS["results"], 
                                      f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            
            self.logger.info(f"Evaluation completed. Results saved to {results_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed with error: {str(e)}")
            raise

    def compare_to_benchmark(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare model performance to benchmark (e.g., S&P 500).
        
        Args:
            model_results: Results from model evaluation
            
        Returns:
            Dictionary containing comparison metrics
        """
        try:
            
            # Download benchmark data
            benchmark_data = yf.download(
                self.config["benchmark_symbol"],
                start=(datetime.now() - pd.Timedelta(days=self.config["evaluation_period_days"])),
                end=datetime.now()
            )
            
            # Calculate benchmark returns
            benchmark_returns = benchmark_data['Close'].pct_change().dropna().values
            benchmark_metrics = self.calculate_metrics(benchmark_returns)
            
            # Calculate relative metrics
            comparison = {
                "relative_sharpe": model_results["metrics"]["sharpe_ratio"] - benchmark_metrics["sharpe_ratio"],
                "relative_return": model_results["metrics"]["total_return"] - benchmark_metrics["total_return"],
                "benchmark_metrics": benchmark_metrics
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Benchmark comparison failed with error: {str(e)}")
            return {"error": str(e)}