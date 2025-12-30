# QuantConnect (Lean) and Interactive Brokers (IBKR) Roles

This project can run its own trading logic, but pairing it with QuantConnect's open-source Lean engine and Interactive Brokers' brokerage access helps when you need a professional, recruiter-ready live/paper setup.

## Why involve Lean at all?
- **Execution orchestrator**: Lean handles the runtime loop (data intake, signal evaluation, order submission) and keeps the code path identical across backtests, paper, and live.
- **Market data plumbing**: Lean requests live quotes/option chains from your connected broker, then feeds them into your strategy.
- **Broker abstraction**: You can swap brokers (IBKR, OANDA, Tradier, Alpaca, etc.) without rewriting strategy code, relying on Lean's brokerage plugins.
- **Reporting**: Lean generates standardized reports/tear sheets that recruiters recognize.

## Where IBKR fits
- **Broker & venue**: IBKR is the account that holds cash, enforces margin, and routes orders to exchanges.
- **Market access**: Provides real-time quotes and order acknowledgements (paper and live share the same API).
- **Options support**: U.S. equity options chains, Greeks, and multi-leg/combination orders are available through Lean's IBKR plugin.

## Data vs. execution responsibilities
- **Lean**: Requests current prices/option chains via the broker connection, evaluates your strategy rules, and issues orders.
- **IBKR**: Receives those orders, applies margin and market checks, and executes (paper or live).

## Paper-first workflow
1. Configure Lean to use **IBKR Paper** credentials.
2. Run your strategy unchanged; Lean streams quotes/option data from IBKR Paper and routes orders back.
3. Export Lean reports and IBKR paper fills for your portfolio case study. Swapping to live later is just a credential change.

### Fastest sanity check (no QuantConnect cloud required)
- Start **IBKR Gateway/TWS** in paper mode (default host/port 127.0.0.1:4002).
- Run the local smoke test (submits no trades by default):
  ```bash
  python scripts/run_ibkr_session.py --config config/ibkr.paper.example.yaml --dry-run
  ```
  This validates connectivity, pulls account/portfolio snapshots, and exits. Use `--place-test-order` (see script help) only after confirming paper connectivity.
- To run through **Lean** locally in paper mode (still paper account, no real capital):
  ```bash
  lean live "ibkr-paper" --algorithm-file integrations/lean_ibkr/TradingAgentAlgorithm.py --environment paper
  ```
  This uses your IBKR paper credentials and QuantConnect API token for Lean CLI auth; it does **not** touch QuantConnect cloud live trading unless you explicitly target it.

## Step-by-step setup plan (Lean + IBKR)
1) **Accounts & entitlements**
   - Create/enable **IBKR Paper** (IBKR Lite/Pro with paper credentials) and ensure market data subscriptions for the assets you will trade (equity options if applicable).
   - Install IBKR Gateway/TWS and note the host/port/API settings (default 127.0.0.1:4002 for paper).
   - Sign up for a **QuantConnect account** and generate an **API token** (Lean CLI uses it for auth even when running locally; orders still route to your IBKR account).
   - Confirm Docker is available for Lean runners; if you use QuantConnect cloud later, ensure your QuantConnect account has brokerage permissions for IBKR.

2) **Install Lean locally**
   - Clone the Lean repo or install the Lean CLI (`pip install quantconnect-lean`).
   - Confirm Docker is available (Lean uses containers for brokerage/data runners).

3) **Configure the IBKR brokerage plugin**
   - In your Lean configuration, set `ib-account`, `ib-host`, `ib-port`, and paper credentials.
   - Enable the **paper** flag in the brokerage config; keep live credentials empty until you are ready.

4) **Connect strategy code**
   - Point Lean to your algorithm file (see `integrations/lean_ibkr/TradingAgentAlgorithm.py` as a starter; it reads signals from `metrics/signals/latest.csv`).
   - Add risk limits in Lean config (max position size, daily loss limit) and a “cancel all” kill switch.

5) **Run paper session**
   - Start IBKR Gateway/TWS in paper mode and validate credentials with `scripts/run_ibkr_session.py --config config/ibkr.paper.example.yaml`.
   - Use Lean CLI to launch a paper deployment; validate connectivity before market open.
   - Collect logs, order/fill reports, and the Lean report (tear sheet) for your portfolio.

6) **Optional: switch to live (micro-size)**
   - Swap to live credentials in the Lean brokerage config; keep tight sizing and loss caps.
   - Run for a short interval and capture fills/screenshots for credibility.

## Backtesting with historical data (alternative path)
- You can continue using **this repository’s built-in backtesting pipeline** with Alpha Vantage/Yahoo Finance historical data when Lean is not required.
- To backtest inside Lean instead, supply historical data (QuantConnect data subscriptions or your own) and run Lean in backtest mode using the same algorithm file. This keeps parity with the paper/live config while avoiding broker connectivity.

## When to rely solely on this repository
- For **research, backtests, or simulations** where you control data ingestion (e.g., Alpha Vantage/Yahoo Finance here), your existing pipeline is sufficient.
- For **broker-grade execution and recruiter credibility**, Lean + IBKR provides audited connectivity, option market depth, and familiar tear sheets.
