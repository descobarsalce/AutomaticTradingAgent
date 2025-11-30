# Lean + IBKR Integration Blueprint

This folder provides a runnable scaffold to pair the trading agent with
QuantConnect's open-source Lean engine and Interactive Brokers (paper or live).
Use it as a starting point for recruiter-ready demos with parity between
backtests, paper, and micro-sized live sessions.

## Contents
- `lean.paper.example.json` – template for Lean CLI configuration targeting IBKR Paper.
- `TradingAgentAlgorithm.py` – minimal Lean algorithm that reads signals from
  `metrics/signals/latest.csv` and routes orders through IBKR.

## Setup (paper first)
1. Install Lean CLI: `pip install quantconnect-lean` and ensure Docker is running.
2. Make sure you have:
   - **IBKR Paper** account credentials (and market data for the assets you trade) plus **IBKR Gateway/TWS** running in paper mode (default 127.0.0.1:4002).
   - A **QuantConnect account** with an **API token** (used by the Lean CLI for auth even in local runs; orders still route to your IBKR account).
3. Copy `lean.paper.example.json` to `lean.json` and fill placeholders:
   - `organization-id` and `api-token` from your QuantConnect account.
   - IBKR Paper credentials (`ib-account`, `ib-user-name`, `ib-password`).
   - `ib-host`/`ib-port` matching IBKR Gateway/TWS in paper mode (default 127.0.0.1:4002).
4. Place `TradingAgentAlgorithm.py` in your Lean project directory (or symlink it).
5. Generate signals for your symbols into `metrics/signals/latest.csv` in the repo root
   (format: `SYMBOL,SIGNAL` where SIGNAL is -1, 0, or 1). Wire your model inference
   pipeline to update this file.
6. Run a paper deployment with Lean CLI, e.g.:
   ```bash
   lean live "ibkr-paper" --algorithm-file integrations/lean_ibkr/TradingAgentAlgorithm.py
   ```
7. Collect Lean logs, reports, and IBKR paper fills as portfolio artifacts.

### Quick local smoke test (no QuantConnect cloud deployment)
- Bring up IBKR Gateway/TWS in **paper** mode (default 127.0.0.1:4002).
- Validate credentials/connectivity without placing trades:
  ```bash
  python scripts/run_ibkr_session.py --config config/ibkr.paper.example.yaml --dry-run
  ```
- When ready to test order flow in paper, rerun with `--place-test-order` (see script help) using minimal size.

### Does this connect directly to my QuantConnect account?
- The Lean CLI uses your QuantConnect **API token** for authentication, but paper/live trading still routes to **your IBKR account** (paper or live) via the brokerage plugin.
- By default, the provided `lean.paper.example.json` targets **local Lean paper runs**, not QuantConnect’s managed cloud. You must explicitly configure a cloud deployment if you want to run on QuantConnect’s infrastructure.

## Switching to live (micro-size)
- Duplicate the environment block in `lean.json`, set `environment` to `live`, and
  swap to live credentials/ports (typically 7497). Keep position sizing and daily
  loss caps tight while validating.

## Alternative: direct IBKR adapter
If you prefer to connect without Lean, use `core/brokers/ibkr_adapter.py` and
`scripts/run_ibkr_session.py` with `config/ibkr.paper.example.yaml` to validate
connectivity and submit small test orders before handing control to Lean.
