
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from utils.common import validate_numeric, round_price, MAX_POSITION_SIZE, MIN_POSITION_SIZE

logger = logging.getLogger(__name__)

class PortfolioManager:
    def __init__(self, initial_balance: float, transaction_cost: float = 0.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Portfolio state
        self.positions: Dict[str, float] = {}
        self.position_values: Dict[str, float] = {}
        self.cost_bases: Dict[str, float] = {}
        self.unrealized_pnl: Dict[str, float] = {}
        self.realized_pnl: Dict[str, float] = {}
        
        # Transaction history
        self.trades_history: List[Dict] = []
        self.portfolio_value_history: List[float] = []
        self.cash_history: List[float] = []
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.peak_value = initial_balance
        
    def execute_trade(self, symbol: str, action: int, quantity: float, price: float, timestamp: datetime) -> bool:
        """Execute a trade and update portfolio state."""
        if not validate_numeric(quantity) or not validate_numeric(price):
            return False
            
        is_buy = action == 1
        total_cost = (quantity * price) + self.transaction_cost
        
        if is_buy and total_cost > self.current_balance:
            logger.warning(f"Insufficient funds for trade: {total_cost} > {self.current_balance}")
            return False
            
        if symbol not in self.positions:
            self.positions[symbol] = 0
            self.position_values[symbol] = 0
            self.cost_bases[symbol] = 0
            self.unrealized_pnl[symbol] = 0
            self.realized_pnl[symbol] = 0
        
        # Update position
        old_position = self.positions[symbol]
        if is_buy:
            new_position = old_position + quantity
            self.current_balance -= total_cost
            # Update cost basis
            old_cost = self.cost_bases[symbol] * old_position
            new_cost = price * quantity
            self.cost_bases[symbol] = (old_cost + new_cost) / new_position if new_position > 0 else 0
        else:
            new_position = old_position - quantity
            self.current_balance += (quantity * price) - self.transaction_cost
            # Calculate realized PnL
            realized_pnl = (price - self.cost_bases[symbol]) * quantity
            self.realized_pnl[symbol] += realized_pnl
            
        self.positions[symbol] = round(new_position, 4)
        self.position_values[symbol] = self.positions[symbol] * price
        
        # Record trade
        self.trades_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'BUY' if is_buy else 'SELL',
            'quantity': quantity,
            'price': price,
            'cost_basis': self.cost_bases[symbol],
            'total_cost': total_cost,
            'balance_after': self.current_balance
        })
        
        self._update_metrics(price, symbol)
        return True
        
    def _update_metrics(self, current_price: float, symbol: str) -> None:
        """Update portfolio metrics."""
        # Update unrealized PnL
        if self.positions[symbol] != 0:
            self.unrealized_pnl[symbol] = (current_price - self.cost_bases[symbol]) * self.positions[symbol]
            
        # Update portfolio value and track history
        total_value = self.get_total_value()
        self.portfolio_value_history.append(total_value)
        self.cash_history.append(self.current_balance)
        
        # Update maximum drawdown using the same method as MetricsCalculator
        if len(self.portfolio_value_history) > 1:
            rolling_max = max(self.portfolio_value_history)
            current_drawdown = (rolling_max - total_value) / rolling_max
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
    def get_total_value(self) -> float:
        """Get total portfolio value including cash and positions."""
        return self.current_balance + sum(self.position_values.values())
        
    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of current positions."""
        data = []
        for symbol in self.positions.keys():
            data.append({
                'symbol': symbol,
                'position': self.positions[symbol],
                'value': self.position_values[symbol],
                'cost_basis': self.cost_bases[symbol],
                'unrealized_pnl': self.unrealized_pnl[symbol],
                'realized_pnl': self.realized_pnl[symbol]
            })
        return pd.DataFrame(data)
        
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        return pd.DataFrame(self.trades_history)
        
    def get_portfolio_metrics(self) -> Dict:
        """Get key portfolio metrics."""
        returns = np.diff(self.portfolio_value_history) / self.portfolio_value_history[:-1]
        return {
            'total_value': self.get_total_value(),
            'cash_balance': self.current_balance,
            'total_pnl': sum(self.realized_pnl.values()) + sum(self.unrealized_pnl.values()),
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': np.mean(returns) / np.std(returns) if len(returns) > 0 else 0,
            'total_trades': len(self.trades_history)
        }
