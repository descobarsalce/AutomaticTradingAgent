from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm
import time
from typing import Optional

class ProgressBarCallback(BaseCallback):
    """
    Custom callback for adding a progress bar with percentage and time estimation during training
    """
    def __init__(self, total_timesteps: int, progress_bar=None, status_placeholder=None):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.start_time = None
        self.progress_bar = progress_bar
        self.status_placeholder = status_placeholder
        
    def _on_training_start(self):
        self.start_time = time.time()
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self):
        self.pbar.update(1)
        
        if self.progress_bar is not None and self.status_placeholder is not None:
            current_step = self.num_timesteps
            progress = current_step / self.total_timesteps
            self.progress_bar.progress(progress)
            
            # Calculate time estimation
            elapsed_time = time.time() - self.start_time
            if current_step > 0:
                estimated_total_time = elapsed_time * (self.total_timesteps / current_step)
                remaining_time = estimated_total_time - elapsed_time
                
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
        return True
        
    def _on_training_end(self):
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
