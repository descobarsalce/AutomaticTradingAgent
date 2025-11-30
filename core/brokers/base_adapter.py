"""Abstract broker adapter interface for live/paper trading.

This module defines the minimal surface that concrete broker adapters must
implement to let the trading agent submit orders, retrieve positions, and
inspect account state. It is deliberately simple so strategies can swap
brokers (IBKR, Alpaca, OANDA, etc.) without rewriting core logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


@dataclass
class OrderRequest:
    """Normalized order request used by broker adapters."""

    symbol: str
    quantity: float
    side: str  # "BUY" or "SELL"
    order_type: str = "MKT"  # "MKT" or "LMT"
    limit_price: Optional[float] = None
    time_in_force: str = "DAY"
    security_type: str = "STK"  # "STK" or "OPT"
    exchange: str = "SMART"
    currency: str = "USD"
    expiry: Optional[str] = None  # YYYYMMDD for options
    strike: Optional[float] = None
    right: Optional[str] = None  # "C" or "P" for calls/puts
    multiplier: Optional[str] = None


class BrokerAdapter(ABC):
    """Base interface for broker adapters."""

    @abstractmethod
    def connect(self) -> None:
        """Establish a session with the broker (paper or live)."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the session and release resources."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return whether the session is healthy."""

    @abstractmethod
    def submit_order(self, order: OrderRequest) -> str:
        """Submit an order and return a broker-specific identifier."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order by identifier."""

    @abstractmethod
    def get_positions(self) -> Iterable[Dict[str, Any]]:
        """Return open positions with quantities and average cost."""

    @abstractmethod
    def get_account_summary(self) -> Dict[str, Any]:
        """Return high-level account metrics (cash, buying power, etc.)."""
