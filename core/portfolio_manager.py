import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from utils.common import validate_numeric, round_price, MAX_POSITION_SIZE, MIN_POSITION_SIZE
from metrics.metric_sink import MetricsSink
from metrics.metrics_calculator import MetricsCalculator
import random

logger = logging.getLogger(__name__)

high_verbosity = False


class RiskBudgetManager:
    """Controls risk scaling based on forward-looking signals."""

    def __init__(
        self,
        target_volatility: float = 0.2,
        max_scale: float = 1.5,
        min_scale: float = 0.1,
        corr_sensitivity: float = 0.5,
        drawdown_sensitivity: float = 1.0,
        per_asset_cap: float = 0.35,
        max_leverage: float = 1.0,
    ) -> None:
        self.target_volatility = target_volatility
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.corr_sensitivity = corr_sensitivity
        self.drawdown_sensitivity = drawdown_sensitivity
        self.per_asset_cap = per_asset_cap
        self.max_leverage = max_leverage
        self._eps = 1e-6

    def compute_scale(
        self,
        forecast_volatility: float,
        correlation_proxy: float,
        drawdown: float,
    ) -> float:
        """Compute a scalar to down/up-weight raw actions."""

        effective_vol = max(forecast_volatility, self._eps)
        base_scale = self.target_volatility / effective_vol

        risk_penalty = 1 + (self.corr_sensitivity * max(correlation_proxy, 0.0))
        risk_penalty += self.drawdown_sensitivity * max(drawdown, 0.0)

        scaled = base_scale / risk_penalty
        clipped = float(np.clip(scaled, self.min_scale, self.max_scale))
        return clipped

    def apply_risk_budget(
        self,
        raw_weights: np.ndarray,
        forecast_volatility: float,
        correlation_proxy: float,
        drawdown: float,
    ) -> Tuple[np.ndarray, float, float]:
        """Return scaled weights plus the scaling factor and cap used."""

        scale = self.compute_scale(forecast_volatility, correlation_proxy, drawdown)
        weights = np.array(raw_weights, dtype=float) * scale

        if self.per_asset_cap is not None:
            weights = np.clip(weights, -self.per_asset_cap, self.per_asset_cap)

        gross_exposure = float(np.abs(weights).sum())
        if gross_exposure > self.max_leverage:
            weights = weights * (self.max_leverage / gross_exposure)

        return weights, scale, self.per_asset_cap


class PortfolioManager:
    def __init__(self, initial_balance: float, transaction_cost: float = 0.0,
                 risk_budget_params: Optional[Dict[str, float]] = None,
                 metrics_sink: Optional[MetricsSink] = None):  # <-- new parameter
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.transaction_cost = transaction_cost

        self.risk_budget_manager = RiskBudgetManager(**(risk_budget_params or {}))
        self.metrics_sink = metrics_sink or MetricsSink()
        self.risk_ledger: List[Dict] = []

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
        self._trade_history: List[Dict] = []  # track environment-specific history here if desired

        # Risk metrics
        self.max_drawdown = 0.0
        self.peak_value = initial_balance

    def update_position_values(self, price: float, symbol: str):
        self.position_values[symbol] = self.positions[symbol] * price
        
    def _handle_buy(self, symbol: str, quantity: float, price: float, total_cost: float) -> bool:
        """Handle position and balance updates for a BUY."""
        try:
            old_position = self.positions.get(symbol, 0.0)
            new_position = old_position + quantity
            self.current_balance -= total_cost
            old_cost = self.cost_bases.get(symbol, 0.0) * old_position
            new_cost = price * quantity
            self.cost_bases[symbol] = (old_cost + new_cost) / new_position if new_position > 0 else 0
            self.positions[symbol] = new_position  # Changed: use exact position value
            self.update_position_values(price, symbol)
            # Additional check: stop-loss trigger placeholder and position limits
            # e.g., enforce maximum position value per asset
            # if new_position > MAX_POSITION_SIZE:
            #     logger.warning(f"Buy rejected: {symbol} exceeds max allowed position")
            #     return False
            return True
        except Exception as e:
            logger.error(f"Error in _handle_buy: {str(e)}")
            return False

    def _handle_sell(self, symbol: str, quantity: float, price: float) -> bool:
        """Handle position and balance updates for a SELL."""
        try:
            old_position = self.positions.get(symbol, 0.0)
            new_position = old_position - quantity
            self.current_balance += (quantity * price) - self.transaction_cost
            realized_pnl = (price - self.cost_bases[symbol]) * quantity
            self.realized_pnl[symbol] += realized_pnl
            self.positions[symbol] = new_position  # Changed: use exact position value
            self.update_position_values(price, symbol)
            # Additional: trigger stop-loss exit if loss exceeds a threshold
            # if (price - self.cost_bases[symbol]) / self.cost_bases[symbol] < -STOP_LOSS_THRESHOLD:
            #     logger.info(f"Stop-loss triggered for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error in _handle_sell: {str(e)}")
            return False

    def _log_trade_entry(self, trade: Dict) -> None:
        # logger.info(f"Trade Executed - Timestamp: {trade['timestamp']}, Symbol: {trade['symbol']}, "
        #             f"Action: {trade['action']}, Quantity: {trade['quantity']:.4f}, Price: {trade['price']:.4f}, "
        #             f"Cost Basis: {trade['cost_basis']:.4f}, Total Cost: {trade['total_cost']:.4f}, "
        #             f"Balance After: {trade['balance_after']:.4f}")
        pass

    def _centralized_log(self, messages: list) -> None:
        # Logs all messages in a single call to reduce clutter.
        logger.info("\n".join(messages))

    def _group_log(self, messages: list) -> None:
        logger.info("\n".join(messages))

    def execute_trade(self, symbol: str, quantity: float, price: float, timestamp: datetime) -> bool:
        quantity = float(quantity)
        price = float(price)
        trade_type = 'BUY' if quantity > 0 else 'SELL'
        total_cost = (quantity * price) + self.transaction_cost

        if not isinstance(quantity, (int, float)) or not isinstance(price, (int, float)):
            logger.error(f"Trade for {symbol} failed: Invalid quantity or price (quantity={quantity}, price={price})")
            return False

        is_buy = quantity > 0
        if is_buy:
            if total_cost > self.current_balance or (self.current_balance - total_cost) < 0:
                if high_verbosity:
                    logger.warning(f"Buy trade for {symbol} rejected: Insufficient funds.")
                return False
        else:
            sell_qty = abs(quantity)
            prospective_balance = self.current_balance + (sell_qty * price) - self.transaction_cost
            if prospective_balance < 0:
                if high_verbosity:
                    logger.warning(f"Sell trade for {symbol} rejected: Negative balance would result.")
                return False

        # Ensure symbol keys are initialized
        if symbol not in self.positions:
            self.positions[symbol] = 0
            self.position_values[symbol] = 0
            self.cost_bases[symbol] = 0
            self.unrealized_pnl[symbol] = 0
            self.realized_pnl[symbol] = 0

        executed = False
        if is_buy:
            executed = self._handle_buy(symbol, quantity, price, total_cost)
        else:
            executed = self._handle_sell(symbol, abs(quantity), price)

        # Log a single structured summary for the trade execution.
        # logger.info(f"Trade Summary - {symbol}: Order {trade_type}, Qty {quantity}, Price {price}, "
        #             f"Total Cost {total_cost}, After: Balance {self.current_balance}, "
        #             f"Positions {self.positions}, Result: {'Success' if executed else 'Failure'}")

        # Record and log the trade details
        trade_entry = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': trade_type,
            'quantity': quantity,
            'price': price,
            'cost_basis': self.cost_bases[symbol],
            'total_cost': total_cost,
            'balance_after': self.current_balance
        }
        self.trades_history.append(trade_entry)
        self._trade_history.append(trade_entry)

        # Log the trade in a centralized table format (unchanged).
        self._log_trade_entry(trade_entry)

        return executed

    def _update_metrics(self, current_price: float, symbol: str) -> None:
        """Update portfolio metrics."""
        # Update unrealized PnL
        if self.positions[symbol] != 0:
            self.unrealized_pnl[symbol] = (current_price - self.cost_bases[symbol]) * self.positions[symbol]

        # Update portfolio value and track history
        total_value = self.get_total_value()
        self.portfolio_value_history.append(total_value)
        self.cash_history.append(self.current_balance)

        # Update maximum drawdown using MetricsCalculator
        if len(self.portfolio_value_history) > 1:
            self.max_drawdown = MetricsCalculator.calculate_maximum_drawdown(self.portfolio_value_history)

        # Check for NaN values in portfolio metrics
        if np.isnan(total_value) or np.isnan(self.max_drawdown):
            logger.error(f"NaN detected in portfolio metrics: total_value={total_value}, max_drawdown={self.max_drawdown}")
            raise ValueError("NaN detected in portfolio metrics")

        logger.debug(f"Updated metrics: total_value={total_value}, max_drawdown={self.max_drawdown}")

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

    def get_environment_trade_history(self) -> List[Dict]:
        """Get environment-specific trade history."""
        return self._trade_history

    def record_risk_event(
        self,
        timestamp: datetime,
        forecast_volatility: float,
        correlation_proxy: float,
        drawdown: float,
        scale: float,
        per_asset_cap: float,
    ) -> None:
        event = {
            'timestamp': timestamp,
            'forecast_volatility': float(forecast_volatility),
            'correlation_proxy': float(correlation_proxy),
            'drawdown': float(drawdown),
            'scale': float(scale),
            'per_asset_cap': float(per_asset_cap),
        }
        self.risk_ledger.append(event)
        if self.metrics_sink:
            self.metrics_sink.emit("risk_budget", event)

    def get_portfolio_metrics(self) -> Dict:
        """Get key portfolio metrics using MetricsCalculator."""
        returns = MetricsCalculator.calculate_returns(self.portfolio_value_history)
        turnover = self.calculate_turnover()
        return {
            'total_value': self.get_total_value(),
            'cash_balance': self.current_balance,
            'total_pnl': sum(self.realized_pnl.values()) + sum(self.unrealized_pnl.values()),
            'max_drawdown': MetricsCalculator.calculate_maximum_drawdown(self.portfolio_value_history),
            'sharpe_ratio': MetricsCalculator.calculate_sharpe_ratio(returns),
            'sortino_ratio': MetricsCalculator.calculate_sortino_ratio(returns),
            'total_trades': len(self.trades_history),
            'total_return': (self.get_total_value() - self.initial_balance) / self.initial_balance,
            'turnover': turnover
        }

    def calculate_turnover(self) -> float:
        if not self.trades_history:
            return 0.0

        total_traded = sum(
            abs(trade.get('quantity', 0) * trade.get('price', 0))
            for trade in self.trades_history
        )
        if self.portfolio_value_history:
            average_value = float(np.mean(self.portfolio_value_history))
        else:
            average_value = self.initial_balance

        if average_value <= 0:
            return 0.0

        return float(total_traded / average_value)

    def _calculate_trade_quantity(self, action: float, symbol: str, price: float, max_pct_position_by_asset: float) -> float:
        if price <= 0:
            logger.info(f"Invalid price for {symbol}: {price}")
            return 0.0
        scaled_size = abs(action) * max_pct_position_by_asset
        if action > 0:
            if self.current_balance <= 0:
                # logger.info(f"Insufficient funds for buy on {symbol} with balance {self.current_balance}")
                return 0.0
            qty = (self.current_balance * scaled_size) / price
            # logger.info(f"Calculated BUY quantity for {symbol}: Qty {qty} (Max Trade Amount {(self.current_balance * scaled_size)})")
        else:
            current_position = self.positions.get(symbol, 0.0)
            qty = current_position * scaled_size
            # logger.info(f"Calculated SELL quantity for {symbol}: Qty {qty} (Current Position {current_position})")
        return qty

    def execute_all_trades(self, stock_names: List[str], actions: np.ndarray,
                           get_current_price: callable, max_pct_position_by_asset: float,
                           timestamp: 'datetime', risk_controls: Optional[Dict[str, float]] = None) -> Dict[str, bool]:
        risk_controls = risk_controls or {}
        trades_executed = {symbol: False for symbol in stock_names}

        forecast_vol = float(risk_controls.get('forecast_volatility', 0.0))
        corr_proxy = float(risk_controls.get('correlation_proxy', 0.0))
        drawdown = float(risk_controls.get('drawdown', 0.0))

        actions_for_trading, scale, per_asset_cap = self.risk_budget_manager.apply_risk_budget(
            actions, forecast_vol, corr_proxy, drawdown)
        self.record_risk_event(timestamp, forecast_vol, corr_proxy, drawdown, scale, per_asset_cap)

        # First process all sell actions
        for symbol, action in zip(stock_names, actions_for_trading):
            if action < 0:  # Only process sells first
                price = get_current_price(symbol, open_price=True)
                qty = self._calculate_trade_quantity(action, symbol, price, max_pct_position_by_asset)
                if qty > 0:
                    if self.execute_trade(symbol, -qty, price, timestamp):  # Note the negative qty for sells
                        trades_executed[symbol] = True

                        self._update_metrics(price, symbol)
                    else:
                        if high_verbosity:
                            logger.error(f"Sell trade for {symbol} failed in execution.")

        # Then process all buy actions (randomize order to reduce bias)
        pairs = list(zip(stock_names, actions_for_trading))
        random.shuffle(pairs)
        for symbol, action in pairs:
            if action > 0:  # Only process buys after sells
                price = get_current_price(symbol, open_price=True)
                qty = self._calculate_trade_quantity(action, symbol, price, max_pct_position_by_asset)
                if qty > 0:
                    if self.execute_trade(symbol, qty, price, timestamp):
                        trades_executed[symbol] = True
                        self._update_metrics(price, symbol)
                    else:
                        if high_verbosity:
                            logger.error(f"Buy trade for {symbol} failed in execution.")

        return trades_executed
