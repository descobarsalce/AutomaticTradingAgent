"""
Centralized configuration management for the trading system.
All parameters for training, optimization, and testing are stored here.
"""
import os
from typing import Dict, Any, List
import torch.nn as nn

# Training Configuration
TRAINING_CONFIG: Dict[str, Any] = {
    "initial_balance": 10000,
    "transaction_cost": 0.01,
    "portfolio_action_scheme": "discrete",
    "max_position_size": 1.0,
    "learning_rate": 0.001,
    "batch_size": 64,
    "n_steps": 2048,
    "total_timesteps": 100000,
    "train_test_split": 0.8,
    "reward_scaling": 1.0,
}

# Optimization Configuration
OPTIMIZATION_CONFIG: Dict[str, Any] = {
    "n_trials": 20,
    "metric": "sharpe_ratio",
    "pruning_enabled": True,
    "study_name": "trading_optimization",
    "optimization_metrics": ["sharpe_ratio", "sortino_ratio", "max_drawdown"],
    "parameter_space": {
        "learning_rate": (1e-5, 1e-2, "log"),
        "batch_size": (32, 256, "int"),
        "n_steps": (1024, 4096, "int"),
    }
}

# Testing Configuration
TESTING_CONFIG: Dict[str, Any] = {
    "evaluation_period_days": 365,
    "metrics": [
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "calmar_ratio"
    ],
    "benchmark_symbol": "SPY",  # S&P 500 ETF as benchmark
    "risk_free_rate": 0.02,  # Annual risk-free rate for Sharpe ratio calculation
}

# Environment Configuration
ENV_CONFIG: Dict[str, Any] = {
    "window_size": 30,
    "features": [
        "close",
        "volume",
        "rsi_14",
        "macd",
        "bollinger_upper",
        "bollinger_lower"
    ],
    "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
    "data_source": "yfinance",
}

# Model Configuration
MODEL_CONFIG: Dict[str, Any] = {
    "policy": "MlpPolicy",
    "verbose": 1,
    "tensorboard_log": "./tensorboard_logs",
    "device": "auto",  # Will use GPU if available
    "policy_kwargs": {
        "net_arch": [128, 128],
        "activation_fn": nn.Tanh  # Using actual PyTorch activation function instead of string
    }
}

# Paths Configuration
PATHS = {
    "models": "./models",
    "data": "./data",
    "logs": "./eval_logs",
    "results": "./results",
    "tensorboard": "./tensorboard_logs",
}

# Create directories if they don't exist
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)