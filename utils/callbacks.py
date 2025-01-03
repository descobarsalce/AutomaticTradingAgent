from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm
import time
import numpy as np
from typing import Optional, Dict, Any
import logging

# Configure logger
logger = logging.getLogger(__name__)

class PortfolioMetricsCallback(BaseCallback):
    def __init__(self, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.portfolio_history: list = []
        self.returns_history: list = []
        self.metrics_history: Dict[str, list] = {
            'sharpe_ratio': [],
            'sortino_ratio': [],
            'max_drawdown': [],
            'returns': []
        }

    def _on_step(self) -> bool:
        try:
            info = self.training_env.get_attr('_last_info')[0]
            if info and isinstance(info, dict) and 'net_worth' in info:
                self.portfolio_history.append(info['net_worth'])

                if self.n_calls % self.eval_freq == 0:
                    if len(self.portfolio_history) > 1:
                        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
                        self.returns_history.extend(returns.tolist())

                        if len(returns) > 0:
                            self.metrics_history['returns'].append(float(np.mean(returns)))
                            logger.info(f"Training step {self.n_calls}: Average return = {float(np.mean(returns)):.4f}")
        except Exception as e:
            logger.error(f"Error in PortfolioMetricsCallback: {str(e)}")
        return True

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps: int, progress_bar=None, status_placeholder=None):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.progress_bar = progress_bar
        self.status_placeholder = status_placeholder
        self.last_update_time = 0
        self.update_interval = 0.5  # Update UI every 0.5 seconds

    def _on_training_start(self):
        try:
            self.start_time = time.time()
            self.last_update_time = self.start_time
            logger.info("Training started")
            if not self.progress_bar:
                self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")
        except Exception as e:
            logger.error(f"Error in _on_training_start: {str(e)}")

    def _on_step(self) -> bool:
        try:
            current_time = time.time()

            # Update console progress bar
            if self.pbar:
                self.pbar.update(1)

            # Update Streamlit UI less frequently to avoid overhead
            if (current_time - self.last_update_time) >= self.update_interval:
                if self.progress_bar is not None and self.status_placeholder is not None:
                    current_step = min(self.num_timesteps, self.total_timesteps)
                    progress = min(1.0, current_step / self.total_timesteps)
                    self.progress_bar.progress(progress)

                    elapsed_time = current_time - self.start_time
                    if current_step > 0:
                        estimated_total_time = elapsed_time * (self.total_timesteps / current_step)
                        remaining_time = max(0, estimated_total_time - elapsed_time)

                        if remaining_time < 60:
                            time_str = f"{remaining_time:.0f} seconds"
                        elif remaining_time < 3600:
                            time_str = f"{remaining_time/60:.1f} minutes"
                        else:
                            time_str = f"{remaining_time/3600:.1f} hours"

                        self.status_placeholder.text(
                            f"Training Progress: {progress*100:.1f}% complete\n"
                            f"Steps: {current_step}/{self.total_timesteps}\n"
                            f"Estimated time remaining: {time_str}"
                        )

                self.last_update_time = current_time

            return True

        except Exception as e:
            logger.error(f"Error in _on_step: {str(e)}")
            return True

    def _on_training_end(self):
        try:
            if self.pbar:
                self.pbar.close()
                self.pbar = None
            if self.progress_bar is not None:
                self.progress_bar.progress(1.0)
            if self.status_placeholder is not None:
                total_time = time.time() - self.start_time
                if total_time < 60:
                    time_str = f"{total_time:.0f} seconds"
                elif total_time < 3600:
                    time_str = f"{total_time/60:.1f} minutes"
                else:
                    time_str = f"{total_time/3600:.1f} hours"
                self.status_placeholder.text(f"Training completed in {time_str}")
                logger.info(f"Training completed in {time_str}")

        except Exception as e:
            logger.error(f"Error in _on_training_end: {str(e)}")