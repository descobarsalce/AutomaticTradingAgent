from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm
import time
import numpy as np
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps: int, progress_bar=None, status_placeholder=None):
        super().__init__(verbose=1)
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.progress_bar = progress_bar
        self.status_placeholder = status_placeholder
        logger.info(f"Initializing ProgressBarCallback with {total_timesteps} total timesteps")

    def _on_training_start(self):
        """Initialize progress tracking at start of training."""
        try:
            self.start_time = time.time()
            if not self.progress_bar:  # If no Streamlit progress bar, use tqdm
                self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")
            logger.info("Training started")
        except Exception as e:
            logger.error(f"Error in _on_training_start: {str(e)}", exc_info=True)
            raise

    def _on_step(self):
        """Update progress on each step."""
        try:
            if self.pbar:
                self.pbar.update(1)

            if self.progress_bar is not None and self.status_placeholder is not None:
                current_step = self.num_timesteps
                progress = min(1.0, current_step / self.total_timesteps)
                self.progress_bar.progress(progress)

                # Calculate time estimates
                if current_step > 0 and self.start_time is not None:
                    elapsed_time = time.time() - self.start_time
                    estimated_total_time = elapsed_time * (self.total_timesteps / current_step)
                    remaining_time = max(0, estimated_total_time - elapsed_time)

                    # Format time string
                    if remaining_time < 60:
                        time_str = f"{remaining_time:.0f} seconds"
                    elif remaining_time < 3600:
                        time_str = f"{remaining_time/60:.1f} minutes"
                    else:
                        time_str = f"{remaining_time/3600:.1f} hours"

                    self.status_placeholder.text(
                        f"Training Progress: {progress*100:.1f}% complete\n"
                        f"Estimated time remaining: {time_str}"
                    )

                    # Log progress at certain intervals
                    if current_step % max(1, self.total_timesteps // 10) == 0:
                        logger.info(f"Training progress: {progress*100:.1f}% complete")

            return True

        except Exception as e:
            logger.error(f"Error in _on_step: {str(e)}", exc_info=True)
            return False

    def _on_training_end(self):
        """Clean up and show final statistics."""
        try:
            if self.pbar:
                self.pbar.close()
                self.pbar = None

            if self.progress_bar is not None:
                self.progress_bar.progress(1.0)

            if self.status_placeholder is not None and self.start_time is not None:
                total_time = time.time() - self.start_time
                if total_time < 60:
                    time_str = f"{total_time:.0f} seconds"
                elif total_time < 3600:
                    time_str = f"{total_time/60:.1f} minutes"
                else:
                    time_str = f"{total_time/3600:.1f} hours"

                self.status_placeholder.text(f"Training completed in {time_str}")

            logger.info("Training completed successfully")

        except Exception as e:
            logger.error(f"Error in _on_training_end: {str(e)}", exc_info=True)

class PortfolioMetricsCallback(BaseCallback):
    """Callback for tracking portfolio metrics during training."""
    def __init__(self, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.portfolio_history = []
        self.returns_history = []
        self.metrics_history: Dict[str, list] = {
            'sharpe_ratio': [],
            'sortino_ratio': [],
            'max_drawdown': [],
            'returns': []
        }
        logger.info("Initialized PortfolioMetricsCallback")

    def _on_step(self) -> bool:
        try:
            info = self.training_env.get_attr('_last_info')[0]
            if info and 'net_worth' in info:
                self.portfolio_history.append(info['net_worth'])

                if self.n_calls % self.eval_freq == 0:
                    if len(self.portfolio_history) > 1:
                        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
                        self.returns_history.extend(returns)

                        if len(returns) > 0:
                            self.metrics_history['returns'].append(float(np.mean(returns)))

                        logger.debug(f"Updated portfolio metrics at step {self.n_calls}")

            return True

        except Exception as e:
            logger.error(f"Error in PortfolioMetricsCallback: {str(e)}", exc_info=True)
            return True  # Continue training despite metrics error