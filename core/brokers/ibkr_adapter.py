"""Interactive Brokers adapter built on ib_insync.

This adapter covers the minimum viable operations needed for paper/live
sessions in IBKR: connecting, submitting market/limit orders (stocks or
options), cancelling, and pulling account/position snapshots. It is designed
to be used directly or from a Lean integration where IBKR is the execution
venue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

from core.brokers.base_adapter import BrokerAdapter, OrderRequest

try:  # ib_insync is optional until a user enables IBKR connectivity
    from ib_insync import IB, Contract, Option, Stock, MarketOrder, LimitOrder
except ImportError as exc:  # pragma: no cover - dependency not always installed
    raise ImportError(
        "ib_insync is required for IBKR connectivity. Install with `pip install ib-insync`."
    ) from exc


@dataclass
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = 4002  # 4002 = paper, 7497 = live (TWS defaults)
    client_id: int = 1
    account: Optional[str] = None
    readonly: bool = False
    connect_timeout: int = 10
    auto_disconnect: bool = True
    log_level: int = logging.INFO
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger("ibkr")
        self._logger.setLevel(self.log_level)


class IBKRAdapter(BrokerAdapter):
    """Concrete broker adapter for Interactive Brokers."""

    def __init__(self, config: IBKRConfig) -> None:
        self.config = config
        self.ib = IB()
        self._active_orders: Dict[str, Any] = {}
        self.config._logger.debug("IBKRAdapter initialized with %s", self.config)

    def connect(self) -> None:
        self.config._logger.info(
            "Connecting to IBKR at %s:%s (clientId=%s)",
            self.config.host,
            self.config.port,
            self.config.client_id,
        )
        self.ib.connect(
            self.config.host,
            self.config.port,
            clientId=self.config.client_id,
            readonly=self.config.readonly,
            timeout=self.config.connect_timeout,
        )
        self.config._logger.info("IBKR connection established: %s", self.ib.isConnected())

    def disconnect(self) -> None:
        self.config._logger.info("Disconnecting IBKR session")
        self.ib.disconnect()

    def is_connected(self) -> bool:
        return self.ib.isConnected()

    def _build_contract(self, order: OrderRequest) -> Contract:
        if order.security_type == "OPT":
            if not (order.expiry and order.strike and order.right):
                raise ValueError("Options require expiry (YYYYMMDD), strike, and right ('C' or 'P')")
            contract: Contract = Option(
                symbol=order.symbol,
                lastTradeDateOrContractMonth=order.expiry,
                strike=order.strike,
                right=order.right,
                exchange=order.exchange,
                currency=order.currency,
                multiplier=order.multiplier or "100",
            )
        else:
            contract = Stock(order.symbol, order.exchange, order.currency)
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            raise RuntimeError(f"Unable to qualify contract for {order.symbol}")
        return qualified[0]

    def submit_order(self, order: OrderRequest) -> str:
        contract = self._build_contract(order)
        ib_order = self._build_order(order)
        trade = self.ib.placeOrder(contract, ib_order)
        trade_id = str(trade.order.orderId)
        self._active_orders[trade_id] = trade
        self.config._logger.info(
            "Submitted %s %s x%s as %s (id=%s)",
            order.side,
            order.symbol,
            order.quantity,
            order.order_type,
            trade_id,
        )
        return trade_id

    def _build_order(self, order: OrderRequest):
        quantity = abs(order.quantity)
        is_buy = order.side.upper() == "BUY"
        if order.order_type.upper() == "LMT":
            if order.limit_price is None:
                raise ValueError("Limit orders require a limit_price")
            return LimitOrder(
                action="BUY" if is_buy else "SELL",
                totalQuantity=quantity,
                lmtPrice=order.limit_price,
                tif=order.time_in_force,
            )
        return MarketOrder(action="BUY" if is_buy else "SELL", totalQuantity=quantity, tif=order.time_in_force)

    def cancel_order(self, order_id: str) -> None:
        trade = self._active_orders.get(order_id)
        if not trade:
            raise KeyError(f"No active order with id {order_id}")
        self.ib.cancelOrder(trade.order)
        self.config._logger.info("Cancelled order %s", order_id)

    def get_positions(self) -> Iterable[Dict[str, Any]]:
        positions = []
        for pos in self.ib.positions():
            positions.append(
                {
                    "symbol": pos.contract.symbol,
                    "quantity": float(pos.position),
                    "avg_cost": float(pos.avgCost),
                    "account": pos.account,
                }
            )
        return positions

    def get_account_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        tags = ["TotalCashValue", "BuyingPower", "NetLiquidation", "GrossPositionValue"]
        account_summary = self.ib.accountSummary(
            account=self.config.account or "", tags=",".join(tags)
        )
        for tag in tags:
            matching_value = next(
                (
                    item
                    for item in account_summary
                    if item.tag == tag
                    and (not self.config.account or item.account == self.config.account)
                ),
                None,
            )
            if matching_value:
                summary[tag] = matching_value.value
        if self.config.account:
            summary["account"] = self.config.account
        return summary

    def __enter__(self) -> "IBKRAdapter":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.config.auto_disconnect:
            self.disconnect()
