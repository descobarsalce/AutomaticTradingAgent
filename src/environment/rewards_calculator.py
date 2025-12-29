
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class RewardsCalculator:
    def __init__(self, use_position_profit: bool = False,
                 use_holding_bonus: bool = False,
                 use_trading_penalty: bool = False):
        self.use_position_profit = use_position_profit
        self.use_holding_bonus = use_holding_bonus
        self.use_trading_penalty = use_trading_penalty

    def compute_reward(self, portfolio_history: List[float], 
                      trades_executed: Dict[str, bool],
                      transaction_cost: float = 0.0) -> float:
        """Calculate reward based on portfolio performance and trading behavior."""
        try:
            # Base reward from portfolio value change
            if len(portfolio_history) > 1:
                last_value = portfolio_history[-2]
                current_value = portfolio_history[-1]
                step_return = (current_value - last_value) if last_value > 0 else 0.0
            else:
                step_return = 0.0

            reward = step_return

            # Risk adjustments
            if len(portfolio_history) > 1:
                # Maximum drawdown penalty
                peak = max(portfolio_history)
                if peak > 0:
                    drawdown = (peak - current_value) / peak
                    reward -= drawdown

                # Volatility penalty
                returns = np.diff(portfolio_history) / portfolio_history[:-1]
                volatility = np.std(returns) if len(returns) > 1 else 0
                reward -= volatility

            # Transaction cost penalty
            if self.use_trading_penalty:
                transaction_cost_penalty = sum(trades_executed.values()) * transaction_cost
                reward -= transaction_cost_penalty

            # Clip reward to prevent extreme values
            reward = np.clip(reward, -10, 10)

            if np.isnan(reward):
                logger.error("NaN detected in reward calculation")
                return 0.0

            return float(reward)

        except Exception as e:
            logger.error(f"Error in reward calculation: {str(e)}")
            return 0.0
