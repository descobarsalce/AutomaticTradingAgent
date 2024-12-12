from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm
import time
import numpy as np
from typing import Optional, Dict, Any
from metrics.metrics_calculator import MetricsCalculator

class PortfolioMetricsCallback(BaseCallback):
    """
    Custom callback for tracking portfolio metrics during training
    """
    def __init__(self, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.metrics_calculator = MetricsCalculator()
        self.portfolio_history = []
        self.returns_history = []
        self.metrics_history: Dict[str, list] = {
            'sharpe_ratio': [],
            'sortino_ratio': [],
            'max_drawdown': [],
            'returns': []
        }
        
    def _on_step(self) -> bool:
        """
        Track portfolio metrics at each step
        """
        try:
            # Get current portfolio value from environment info
            info = self.training_env.get_attr('_last_info')[0]
            if info and 'net_worth' in info:
                self.portfolio_history.append(info['net_worth'])
                
                # Calculate metrics every eval_freq steps
                if self.n_calls % self.eval_freq == 0:
                    if len(self.portfolio_history) > 1:
                        # Calculate returns
                        returns = self.metrics_calculator.calculate_returns(self.portfolio_history)
                        if len(returns) > 0:
                            self.returns_history.extend(returns)
                            
                            # Update metrics
                            self.metrics_history['returns'].append(float(np.mean(returns)))
                            self.metrics_history['sharpe_ratio'].append(
                                self.metrics_calculator.calculate_sharpe_ratio(returns)
                            )
                            self.metrics_history['sortino_ratio'].append(
                                self.metrics_calculator.calculate_sortino_ratio(returns)
                            )
                            self.metrics_history['max_drawdown'].append(
                                self.metrics_calculator.calculate_maximum_drawdown(self.portfolio_history)
                            )
                            
                            if self.verbose > 0:
                                latest_metrics = {
                                    k: v[-1] if v else 0.0 
                                    for k, v in self.metrics_history.items()
                                }
                                self.logger.record("train/sharpe_ratio", latest_metrics['sharpe_ratio'])
                                self.logger.record("train/sortino_ratio", latest_metrics['sortino_ratio'])
                                self.logger.record("train/max_drawdown", latest_metrics['max_drawdown'])
                                self.logger.record("train/returns", latest_metrics['returns'])
                                
                                # Update Streamlit metrics display during training
                                try:
                                    import streamlit as st
                                    metrics_cols = st.columns(4)
                                    metrics_cols[0].metric("Sharpe Ratio", f"{latest_metrics['sharpe_ratio']:.2f}")
                                    metrics_cols[1].metric("Sortino Ratio", f"{latest_metrics['sortino_ratio']:.2f}")
                                    metrics_cols[2].metric("Max Drawdown", f"{latest_metrics['max_drawdown']:.2%}")
                                    metrics_cols[3].metric("Average Returns", f"{latest_metrics['returns']:.2%}")
                                except Exception as e:
                                    if self.verbose > 1:
                                        print(f"Error updating metrics display: {str(e)}")
                                
        except Exception as e:
            if self.verbose > 0:
                print(f"Error in PortfolioMetricsCallback: {str(e)}")
                
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the latest metrics"""
        return {
            k: (v[-1] if v else 0.0)
            for k, v in self.metrics_history.items()
        }

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
            current_step = min(self.num_timesteps, self.total_timesteps)
            progress = min(1.0, current_step / self.total_timesteps)
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
