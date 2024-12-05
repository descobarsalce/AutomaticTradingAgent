from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

class ProgressBarCallback(BaseCallback):
    """
    Custom callback for adding a progress bar during training
    """
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
        
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self):
        self.pbar.update(1)
        return True
        
    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()
            self.pbar = None
