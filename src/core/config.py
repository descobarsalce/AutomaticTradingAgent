
"""Configuration settings for the trading agent."""

DEFAULT_PPO_PARAMS = {
    'learning_rate': 3e-4,
    'n_steps': 1024,
    'batch_size': 256,
    'n_epochs': 25,
    'gamma': 0.99,
    'gae_lambda': 0.98,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.8,
    'max_grad_norm': 0.3,
    'use_sde': False,
    'target_kl': 0.05,
    'verbose': 0
}

PARAM_RANGES = {
    'learning_rate': (1e-5, 5e-4),
    'n_steps': (512, 2048),
    'batch_size': (64, 512),
    'n_epochs': (3, 10),
    'gamma': (0.95, 0.999),
    'gae_lambda': (0.9, 0.99),
    'clip_range': (0.0, 0.3),
    'ent_coef': (0.001, 0.02),
    'vf_coef': (0.4, 0.9),
    'max_grad_norm': (0.3, 0.8),
    'target_kl': (0.02, 0.1)
}

DEFAULT_POLICY_KWARGS = {'net_arch': [dict(pi=[64, 64], vf=[64, 64])]}
