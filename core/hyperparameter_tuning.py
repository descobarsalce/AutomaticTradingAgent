"""
Hyperparameter tuning module using Optuna for the trading agent.
"""
import logging
from typing import Dict, Any, Callable
from datetime import datetime
import os
import json

try:
    import optuna
    import numpy as np
except ImportError as e:
    logging.error(f"Failed to import required packages: {str(e)}")
    raise

from config import OPTIMIZATION_CONFIG, PATHS
from core.training import TrainingManager

class HyperparameterOptimizer:
    def __init__(self, env_creator: Callable, config: Dict[str, Any] = None):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            env_creator: Callable that creates a fresh environment instance
            config: Optional configuration override
        """
        self.env_creator = env_creator
        self.config = {**OPTIMIZATION_CONFIG, **(config or {})}
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for optimization process."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{PATHS['logs']}/optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_study(self):
        """Create and configure Optuna study."""
        pruner = optuna.pruners.MedianPruner() if self.config["pruning_enabled"] else None
        study = optuna.create_study(
            study_name=self.config["study_name"],
            direction="maximize",
            pruner=pruner
        )
        return study

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            float: Metric value to be optimized
        """
        # Create a fresh environment for this trial
        env = self.env_creator()
        
        # Sample hyperparameters
        params = {}
        for param_name, (low, high, param_type) in self.config["parameter_space"].items():
            if param_type == "log":
                params[param_name] = trial.suggest_float(param_name, low, high, log=True)
            elif param_type == "int":
                params[param_name] = trial.suggest_int(param_name, low, high)
            else:
                params[param_name] = trial.suggest_float(param_name, low, high)

        # Initialize trainer with sampled parameters
        trainer = TrainingManager(env, config=params)
        trainer.initialize_model()
        
        try:
            # Train the model
            result = trainer.train()
            
            if result["status"] == "failed":
                raise optuna.TrialPruned()
            
            # Calculate the optimization metric
            metric_value = self._calculate_metric(trainer, env)
            
            # Log the trial results
            self.logger.info(f"Trial {trial.number} completed with {self.config['metric']}: {metric_value}")
            
            return metric_value
            
        except Exception as e:
            self.logger.error(f"Trial failed with error: {str(e)}")
            raise optuna.TrialPruned()
        
        finally:
            env.close()

    def _calculate_metric(self, trainer, env) -> float:
        """Calculate the optimization metric for a trained model."""
        # Implement metric calculation based on config["metric"]
        # This is a placeholder - actual implementation would depend on your specific needs
        return 0.0  # Replace with actual metric calculation

    def optimize(self) -> Dict[str, Any]:
        """
        Run the hyperparameter optimization process.
        
        Returns:
            Dict containing optimization results and best parameters
        """
        study = self.create_study()
        
        self.logger.info(f"Starting optimization with {self.config['n_trials']} trials")
        study.optimize(self.objective, n_trials=self.config["n_trials"])
        
        # Get the best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        # Save optimization results
        results = {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(study.trials),
            "optimization_metric": self.config["metric"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results to file
        results_path = os.path.join(PATHS["results"], "optimization_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
            
        self.logger.info(f"Optimization completed. Best {self.config['metric']}: {best_value}")
        self.logger.info(f"Best parameters: {best_params}")
        
        return results