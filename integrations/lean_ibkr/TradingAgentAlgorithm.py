"""Minimal Lean algorithm scaffold that plugs into the IBKR brokerage.

This example keeps parity with the in-repo agent by reading signal files the
agent produces (e.g., in `metrics/signals/latest.csv`) and routing orders
through Lean to IBKR paper/live. Replace the signal loading logic with your
model inference pipeline as needed.
"""

from pathlib import Path

from AlgorithmImports import *  # noqa: F401,F403 - Lean supplies this module


def _load_signal(symbol: str) -> int:
    """Toy signal loader; replace with real model integration.

    Returns -1 for short, 0 for flat, +1 for long.
    """

    try:
        path = Path("metrics/signals/latest.csv")
        if not path.exists():
            return 0
        for line in path.read_text().splitlines():
            parts = line.split(",")
            if len(parts) >= 2 and parts[0] == symbol:
                return int(parts[1])
    except Exception:
        return 0
    return 0


class TradingAgentAlgorithm(QCAlgorithm):
    def Initialize(self) -> None:
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)
        self.symbols = ["SPY"]
        self.sids = {}
        for ticker in self.symbols:
            equity = self.AddEquity(ticker, Resolution.Minute)
            self.sids[ticker] = equity.Symbol
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)
        self.SetWarmUp(timedelta(days=1))

    def OnData(self, slice: Slice) -> None:
        if self.IsWarmingUp:
            return
        for ticker, symbol in self.sids.items():
            signal = _load_signal(ticker)
            quantity = self.CalculateOrderQuantity(symbol, 0.1)  # target 10% allocation
            if signal > 0:
                self.MarketOrder(symbol, abs(quantity))
            elif signal < 0:
                self.MarketOrder(symbol, -abs(quantity))
            else:
                self.Liquidate(symbol)
