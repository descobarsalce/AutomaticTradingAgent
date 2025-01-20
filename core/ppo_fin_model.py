
import numpy as np
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from metrics.metrics_calculator import MetricsCalculator
from environment import SimpleTradingEnv
from core.trading_agent import TradingAgent
from data.data_handler import DataHandler
import tensorflow as tf

class PPOAgentModel:
    def __init__(self):
        self.agent = None
        self.env = None
        self.data_handler = DataHandler()
        self.portfolio_history = []

    def initialize_env(self, data, env_params):
        from utils.common import MIN_TRADE_SIZE
        env_params['min_transaction_size'] = MIN_TRADE_SIZE
        self.env = SimpleTradingEnv(
            data=data,
            initial_balance=env_params['initial_balance'],
            transaction_cost=env_params['transaction_cost'],
            use_position_profit=env_params.get('use_position_profit', False),
            use_holding_bonus=env_params.get('use_holding_bonus', False),
            use_trading_penalty=env_params.get('use_trading_penalty', False),
            training_mode=True)

    def prepare_training_data(self, stock_name: str, start_date: datetime, end_date: datetime):
        portfolio_data = self.data_handler.fetch_data(symbols=[stock_name], start_date=start_date, end_date=end_date)
        if not portfolio_data:
            raise ValueError("No data found in database")
        prepared_data = self.data_handler.prepare_data()
        return next(iter(prepared_data.values()))

    def train(self, stock_name: str, start_date: datetime, end_date: datetime,
              env_params: Dict[str, Any], ppo_params: Dict[str, Any],
              callback=None) -> Dict[str, float]:
        data = self.prepare_training_data(stock_name, start_date, end_date)
        self.initialize_env(data, env_params)
        
        # Configure learning rate schedule
        learning_rate = ppo_params.get('learning_rate', 3e-4)
        decay_steps = ppo_params.get('decay_steps', 1000)
        decay_rate = ppo_params.get('decay_rate', 0.95)
        
        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )
        
        # Update PPO parameters with scheduled learning rate
        ppo_params['learning_rate'] = learning_rate_schedule
        
        # Initialize agent with updated parameters
        self.agent = TradingAgent(env=self.env, ppo_params=ppo_params)

        total_timesteps = (end_date - start_date).days
        self.agent.train(total_timesteps=total_timesteps, callback=callback)
        self.agent.save("trained_model.zip")

        self.portfolio_history = self.env.get_portfolio_history()
        if len(self.portfolio_history) > 1:
            returns = MetricsCalculator.calculate_returns(self.portfolio_history)
            return {
                'sharpe_ratio': MetricsCalculator.calculate_sharpe_ratio(returns),
                'max_drawdown': MetricsCalculator.calculate_maximum_drawdown(self.portfolio_history),
                'sortino_ratio': MetricsCalculator.calculate_sortino_ratio(returns),
                'volatility': MetricsCalculator.calculate_volatility(returns),
                'total_return': (self.portfolio_history[-1] - self.portfolio_history[0]) / self.portfolio_history[0],
                'final_value': self.portfolio_history[-1]
            }
        return {}

    def test(self, stock_name: str, start_date: datetime, end_date: datetime,
             env_params: Dict[str, Any], ppo_params: Dict[str, Any]) -> Dict[str, Any]:
        data = self.prepare_training_data(stock_name, start_date, end_date)
        self.initialize_env(data, env_params)
        self.agent = TradingAgent(env=self.env, ppo_params=ppo_params)
        self.agent.load("trained_model.zip")

        obs, _ = self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        info_history = []

        while not done:
            action = self.agent.predict(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            info_history.append(info)

        portfolio_history = self.env.get_portfolio_history()
        returns = MetricsCalculator.calculate_returns(portfolio_history)

        return {
            'portfolio_history': portfolio_history,
            'returns': returns,
            'info_history': info_history,
            'metrics': {
                'sharpe_ratio': MetricsCalculator.calculate_sharpe_ratio(returns),
                'sortino_ratio': MetricsCalculator.calculate_sortino_ratio(returns),
                'information_ratio': MetricsCalculator.calculate_information_ratio(returns),
                'max_drawdown': MetricsCalculator.calculate_maximum_drawdown(portfolio_history),
                'volatility': MetricsCalculator.calculate_volatility(returns),
                'beta': MetricsCalculator.calculate_beta(returns, data['Close'].pct_change().values)
            }
        }
