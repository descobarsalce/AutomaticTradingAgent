"""
Training module for the reinforcement learning trading agent.
Handles training execution and progress tracking.
"""
import logging
from typing import Dict, Any, Optional
import os
from datetime import datetime
import traceback

try:
    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.utils import set_random_seed
except ImportError as e:
    logging.error(f"Failed to import required packages: {str(e)}")
    raise

from config import TRAINING_CONFIG, MODEL_CONFIG, PATHS

class TrainingManager:
    def __init__(self, env: gym.Env, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the training manager with environment and optional config override.

        Args:
            env: The gymnasium environment for training
            config: Optional configuration override
        """
        self.env = env
        self.config = {**TRAINING_CONFIG, **(config or {})}
        self._ensure_directories()
        self.setup_logging()
        self.model = None

    def _ensure_directories(self):
        """Ensure all required directories exist."""
        required_paths = [
            PATHS['models'],
            PATHS['logs'],
            PATHS['tensorboard'],
            os.path.join(PATHS['models'], 'checkpoints'),
            os.path.join(PATHS['models'], 'best_model')
        ]
        for path in required_paths:
            os.makedirs(path, exist_ok=True)

    def setup_logging(self):
        """Configure logging for training progress."""
        log_file = os.path.join(PATHS['logs'], f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("TrainingManager initialized with configuration: %s", self.config)

    def initialize_model(self):
        """Initialize the PPO model with configured parameters."""
        try:
            self.logger.info("Initializing model with config: %s", MODEL_CONFIG)

            # Set random seed if provided
            if "seed" in self.config:
                set_random_seed(self.config["seed"])

            self.model = PPO(
                policy=MODEL_CONFIG["policy"],
                env=self.env,
                learning_rate=self.config.get("learning_rate", MODEL_CONFIG.get("learning_rate", 0.0003)),
                batch_size=self.config.get("batch_size", MODEL_CONFIG.get("batch_size", 64)),
                n_steps=self.config.get("n_steps", MODEL_CONFIG.get("n_steps", 2048)),
                tensorboard_log=MODEL_CONFIG["tensorboard_log"],
                device=MODEL_CONFIG["device"],
                policy_kwargs=MODEL_CONFIG["policy_kwargs"],
                verbose=0  # Disable built-in progress bar to use custom one
            )
            self.logger.info("Model initialized successfully")

        except Exception as e:
            self.logger.error("Failed to initialize model: %s\n%s", str(e), traceback.format_exc())
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def create_callbacks(self) -> list:
        """Create training callbacks for evaluation and checkpointing."""
        try:
            callbacks = []

            # Evaluation callback
            eval_callback = EvalCallback(
                self.env,
                best_model_save_path=os.path.join(PATHS['models'], 'best_model'),
                log_path=PATHS['logs'],
                eval_freq=self.config.get("eval_freq", 10000),
                deterministic=True,
                render=False,
                verbose=0  # Disable progress output from callback
            )
            callbacks.append(eval_callback)

            # Checkpoint callback
            checkpoint_callback = CheckpointCallback(
                save_freq=self.config.get("save_freq", 50000),
                save_path=os.path.join(PATHS['models'], 'checkpoints'),
                name_prefix="rl_model",
                verbose=0  # Disable progress output from callback
            )
            callbacks.append(checkpoint_callback)

            self.logger.info("Created training callbacks successfully")
            return callbacks

        except Exception as e:
            self.logger.error("Failed to create callbacks: %s\n%s", str(e), traceback.format_exc())
            raise RuntimeError(f"Callback creation failed: {str(e)}")

    def train(self) -> Dict[str, Any]:
        """Execute the training process with configured parameters."""
        if self.model is None:
            self.initialize_model()

        try:
            callbacks = self.create_callbacks()

            self.logger.info("Starting training with total_timesteps: %s", self.config["total_timesteps"])
            self.model.learn(
                total_timesteps=self.config["total_timesteps"],
                callback=callbacks,
                progress_bar=True,
                tb_log_name=f"PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            self.logger.info("Training completed successfully")

            # Save the final model
            final_model_path = os.path.join(PATHS['models'], 'final_model')
            self.model.save(final_model_path)
            self.logger.info("Final model saved to: %s", final_model_path)

            return {
                "status": "success",
                "model_path": final_model_path,
                "config": self.config
            }

        except Exception as e:
            self.logger.error("Training failed: %s\n%s", str(e), traceback.format_exc())
            return {
                "status": "failed",
                "error": str(e),
                "config": self.config
            }

    def save_model(self, path: str):
        """Save the trained model to specified path."""
        try:
            if self.model is not None:
                self.model.save(path)
                self.logger.info("Model saved successfully to: %s", path)
            else:
                self.logger.warning("No model to save")
        except Exception as e:
            self.logger.error("Failed to save model: %s\n%s", str(e), traceback.format_exc())
            raise

    def load_model(self, path: str):
        """Load a trained model from specified path."""
        try:
            self.model = PPO.load(path, env=self.env)
            self.logger.info("Model loaded successfully from: %s", path)
        except Exception as e:
            self.logger.error("Failed to load model: %s\n%s", str(e), traceback.format_exc())
            raise