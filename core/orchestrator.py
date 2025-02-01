"""
Orchestrator module for coordinating training, optimization, and testing components.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import os
import json
import traceback

from config import PATHS
from core.training import TrainingManager
from core.hyperparameter_tuning import HyperparameterOptimizer
from core.testing import ModelTester

class TradingSystemOrchestrator:
    def __init__(self, env_creator, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the trading system orchestrator.

        Args:
            env_creator: Callable that creates fresh environment instances
            config: Optional configuration override
        """
        self.env_creator = env_creator
        self.config = config or {}
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for orchestration process."""
        log_file = os.path.join(PATHS['logs'], f"orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("TradingSystemOrchestrator initialized")

    def run_optimization_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete hyperparameter optimization pipeline.

        Returns:
            Dictionary containing optimization results
        """
        self.logger.info("Starting hyperparameter optimization pipeline")

        try:
            optimizer = HyperparameterOptimizer(self.env_creator, self.config)
            self.logger.info("Created HyperparameterOptimizer instance")

            optimization_results = optimizer.optimize()
            self.logger.info("Optimization completed with results: %s", optimization_results)

            return optimization_results

        except Exception as e:
            self.logger.error("Optimization pipeline failed with error: %s\nTraceback: %s", 
                           str(e), traceback.format_exc())
            raise

    def run_training_pipeline(self, hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline with optional hyperparameters.

        Args:
            hyperparameters: Optional hyperparameters to use for training

        Returns:
            Dictionary containing training results
        """
        self.logger.info("Starting training pipeline with hyperparameters: %s", hyperparameters)

        try:
            env = self.env_creator()
            self.logger.info("Created fresh environment instance")

            trainer = TrainingManager(env, config=hyperparameters)
            self.logger.info("Initialized TrainingManager")

            training_results = trainer.train()
            self.logger.info("Training completed with results: %s", training_results)

            return training_results

        except Exception as e:
            self.logger.error("Training pipeline failed with error: %s\nTraceback: %s", 
                           str(e), traceback.format_exc())
            raise

    def run_testing_pipeline(self, model_path: str) -> Dict[str, Any]:
        """
        Run the complete testing pipeline for a trained model.

        Args:
            model_path: Path to the trained model

        Returns:
            Dictionary containing testing results
        """
        self.logger.info("Starting testing pipeline for model: %s", model_path)

        try:
            env = self.env_creator()
            self.logger.info("Created fresh environment instance")

            tester = ModelTester(env, self.config)
            self.logger.info("Initialized ModelTester")

            # Evaluate model performance
            evaluation_results = tester.evaluate_model(model_path)
            self.logger.info("Model evaluation completed")

            # Compare to benchmark
            benchmark_comparison = tester.compare_to_benchmark(evaluation_results)
            self.logger.info("Benchmark comparison completed")

            results = {
                "evaluation_results": evaluation_results,
                "benchmark_comparison": benchmark_comparison
            }

            self.logger.info("Testing pipeline completed successfully")
            return results

        except Exception as e:
            self.logger.error("Testing pipeline failed with error: %s\nTraceback: %s", 
                           str(e), traceback.format_exc())
            raise

    def run_complete_pipeline(self, optimize_first: bool = True) -> Dict[str, Any]:
        """
        Run the complete pipeline: optimization (optional) -> training -> testing.

        Args:
            optimize_first: Whether to run hyperparameter optimization before training

        Returns:
            Dictionary containing results from all stages
        """
        self.logger.info("Starting complete pipeline execution (optimize_first=%s)", optimize_first)
        try:
            pipeline_results = {}

            # Step 1: Hyperparameter Optimization (if requested)
            if optimize_first:
                self.logger.info("Starting optimization phase")
                optimization_results = self.run_optimization_pipeline()
                pipeline_results["optimization"] = optimization_results
                hyperparameters = optimization_results["best_params"]
                self.logger.info("Optimization completed, best parameters: %s", hyperparameters)
            else:
                hyperparameters = None

            # Step 2: Training
            self.logger.info("Starting training phase")
            training_results = self.run_training_pipeline(hyperparameters)
            pipeline_results["training"] = training_results

            if training_results["status"] == "success":
                # Step 3: Testing
                self.logger.info("Starting testing phase")
                testing_results = self.run_testing_pipeline(training_results["model_path"])
                pipeline_results["testing"] = testing_results
            else:
                self.logger.error("Training failed, skipping testing phase")

            # Save pipeline results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(PATHS["results"], f"pipeline_results_{timestamp}.json")

            with open(results_path, "w") as f:
                json.dump(pipeline_results, f, indent=4)

            self.logger.info("Complete pipeline results saved to %s", results_path)
            return pipeline_results

        except Exception as e:
            self.logger.error("Pipeline execution failed with error: %s\nTraceback: %s", 
                           str(e), traceback.format_exc())
            raise

    def save_state(self, path: str):
        """Save the current state of the trading system."""
        state = {
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
            "paths": PATHS
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=4)

        self.logger.info("System state saved to %s", path)

    def load_state(self, path: str):
        """Load a previously saved state."""
        with open(path, "r") as f:
            state = json.load(f)

        self.config = state["config"]
        self.logger.info("System state loaded from %s", path)