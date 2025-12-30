"""Quickstart script to sanity-check IBKR connectivity.

Usage:
    python scripts/run_ibkr_session.py --config config/ibkr.paper.example.yaml

The script connects to IBKR (paper or live depending on the port), prints a
summary snapshot, and can submit a tiny test order when `--symbol` and
`--quantity` are provided. This is intended as a pre-flight before running the
full agent or handing control to Lean.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from src.core.brokers.base_adapter import OrderRequest
from src.core.brokers.config_loader import load_ibkr_config
from src.core.brokers.ibkr_adapter import IBKRAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IBKR connectivity smoke test")
    parser.add_argument("--config", required=True, help="Path to IBKR YAML config")
    parser.add_argument("--symbol", help="Optional symbol to trade (e.g., AAPL)")
    parser.add_argument("--quantity", type=float, help="Quantity to trade if symbol provided")
    parser.add_argument(
        "--limit", type=float, default=None, help="Limit price (if omitted, market order is used)"
    )
    parser.add_argument(
        "--option-expiry",
        dest="expiry",
        help="Optional option expiry YYYYMMDD (enables option contract mode)",
    )
    parser.add_argument("--option-strike", dest="strike", type=float, help="Option strike")
    parser.add_argument(
        "--option-right", dest="right", choices=["C", "P"], help="Option right (C=call, P=put)"
    )
    parser.add_argument(
        "--paper", action="store_true", help="Alias to indicate paper mode for logging clarity"
    )
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> int:
    parsed = args or parse_args()
    config = load_ibkr_config(parsed.config)
    logging.basicConfig(level=config.log_level)

    with IBKRAdapter(config) as ibkr:
        if not ibkr.is_connected():
            raise SystemExit("Failed to connect to IBKR")

        summary = ibkr.get_account_summary()
        positions = ibkr.get_positions()
        print("Account summary:", summary)
        print("Open positions:", positions)

        if parsed.symbol and parsed.quantity:
            order = OrderRequest(
                symbol=parsed.symbol,
                quantity=parsed.quantity,
                side="BUY" if parsed.quantity > 0 else "SELL",
                order_type="LMT" if parsed.limit else "MKT",
                limit_price=parsed.limit,
                security_type="OPT" if parsed.expiry else "STK",
                expiry=parsed.expiry,
                strike=parsed.strike,
                right=parsed.right,
            )
            order_id = ibkr.submit_order(order)
            print(f"Submitted order id={order_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
