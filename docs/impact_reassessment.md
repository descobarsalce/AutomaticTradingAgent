# Production Impact Reassessment (Data/Feature Plane First, Execution Decoupling Second)

## Updated ranking (highest impact first)

1. **Enforce a unified data/feature plane with SLA-backed inputs**  
   *Why:* The environment still accepts a deprecated `feature_config` and only consumes a provided `feature_pipeline` when callers supply it. That leaves room for training/live skew and missing schema validation for observation tensors. Market data is also sourced from a single provider per run without freshness SLAs or failover logic. Together these gaps make input correctness the dominant production risk.  
   *Evidence:* The Gym environment warns that `feature_config` is deprecated yet continues to rely on it when no pipeline is passed, and it builds observations directly from raw OHLCV when neither path is supplied.【F:environment/trading_env.py†L53-L172】 Trading data ingestion wraps a single provider and validates/aligns OHLCV frames but lacks redundancy, lateness detection, or lineage beyond basic validation.【F:data/data_handler.py†L20-L179】

2. **Decouple inference/execution from the Streamlit UI**  
   *Why:* The UI bootstraps the `UnifiedTradingAgent`, data handler, and session state directly inside Streamlit, meaning the front-end process orchestrates training and any potential inference. This coupling constrains latency, resilience, and deployment separation (no dedicated inference/exec service or message bus). Extracting these concerns into a service layer would materially harden live readiness.  
   *Evidence:* `main.py` initializes `DataHandler` and `UnifiedTradingAgent` in session state during app startup and drives all workflows from tabs within the same Streamlit runtime.【F:main.py†L1-L200】 There is no separate process or API boundary for low-latency inference or order routing.

3. **Expand risk and execution guardrails after inputs and architecture are stable**  
   *Why:* Portfolio checks focus on balance and transaction cost; leverage, concentration, and circuit-breaker controls are present only as commented placeholders. Strengthening these controls is important but should follow input/architecture hardening so policies act on trustworthy data and a resilient execution path.  
   *Evidence:* `PortfolioManager` enforces funding checks but advanced constraints (position caps, stop-loss triggers) are commented out and not enforced, leaving tail-risk gaps until implemented.【F:core/portfolio_manager.py†L15-L175】

## Step-by-step reasoning behind the ordering

1. **Input parity and freshness precede everything else**: Without a mandatory, versioned feature pipeline and multi-source data with SLAs, both training and live inference risk consuming divergent or stale signals. Fixing this first prevents downstream risk controls or execution changes from acting on corrupted inputs.
2. **Service separation reduces operational fragility**: Once inputs are reliable, decoupling inference/execution from the UI addresses latency and availability by isolating critical paths from Streamlit’s interactive runtime. This is the next largest leverage point for production robustness.
3. **Guardrails amplify after foundations are solid**: Richer risk controls have limited value if data/feature integrity or execution architecture remain brittle. After the first two layers are hardened, implementing leverage/concentration/kill-switch logic meaningfully mitigates downside risk.

## Additional production-impact ideas (new)

- **Freshness watchdog and provider failover on the data plane**  
  *Why:* Neither the `TradingEnv` nor `TradingDataManager` tracks feed latency or triggers failover—data is pulled from a single provider instance and trusted if validation passes. Adding latency SLAs, retries, and cached backfill (with manifesting of provider/source) would close a major reliability gap for live inference.  
  *Evidence:* `TradingEnv` requires only a provider object and immediately fetches all OHLCV data with no freshness or redundancy checks.【F:environment/trading_env.py†L55-L101】 `TradingDataManager.fetch` calls a single provider and shifts/validates the frame without lateness detection or multi-provider failover.【F:data/data_handler.py†L20-L45】

- **Schema-locked, versioned feature interface with hard fails on mismatch**  
  *Why:* Observations default to raw OHLCV + positions/balance when no `feature_pipeline` is supplied, and the deprecated `feature_config` path still passes through. Introducing a mandatory, versioned feature contract (shape/dtype ordering) and failing fast on mismatches would prevent silent drift between training and live modes.  
  *Evidence:* The environment warns about deprecated `feature_config` but still relies on it (or unstructured defaults) when a pipeline is absent, constructing observations without schema validation beyond the optional processor.【F:environment/trading_env.py†L89-L172】

- **Queue-based inference/execution service with idempotent order reconciliation**  
  *Why:* Actions are executed synchronously inside the Streamlit-driven process and written directly into `PortfolioManager` state, leaving no boundary for retries, idempotency, or reconciliation against broker fills. A dedicated inference/execution worker consuming a queue (Kafka/Redis) would allow retriable order submission, deduplication, and state sync.  
  *Evidence:* The UI initializes the agent and data handler in Streamlit session state, and trades are applied immediately via `execute_all_trades` without any queuing or reconciliation layer between intent and portfolio state.【F:main.py†L1-L200】【F:core/portfolio_manager.py†L240-L295】

- **Liquidity-aware action sanitizer with ADV/position caps and per-asset sizing**
  *Why:* Current sizing scales linearly with cash or current position and ignores liquidity, spread, or asset-specific caps, so the policy can emit unrealistically large orders. Adding ADV-based clamps, per-symbol max notionals, and spread-aware slippage modeling would yield more realistic fills and reduce blow-up risk.
  *Evidence:* `_calculate_trade_quantity` uses only balance/position and a global percentage cap, without volume, spread, or per-asset liquidity checks before trades are executed.【F:core/portfolio_manager.py†L240-L295】

## First-order expansion ideas (new and non-latency focused)

- **Event-sourced portfolio ledger with crash-safe checkpoints**
  *Why:* Portfolio state is held entirely in-memory (positions, cost bases, trade history) and mutates directly during `execute_trade`, so a process crash would orphan state and make reproducibility impossible. Persisting an append-only ledger with periodic checkpoints would enable deterministic replay, post-mortems, and crash recovery for live trading.
  *Evidence:* `PortfolioManager` maintains positions, balances, and `_trade_history` as plain dictionaries/lists without any persistence or idempotent log; trades are applied immediately when `execute_trade` runs.【F:core/portfolio_manager.py†L15-L124】【F:core/portfolio_manager.py†L166-L214】

- **Calendar- and halt-aware episode gating**
  *Why:* The data manager annotates availability but the environment neither checks market-open status nor halts trades on closed/illiquid sessions—`step` simply advances row-by-row and applies actions. Enforcing market calendars and halts would prevent the agent from “trading” nonexistent sessions and make PnL more realistic.
  *Evidence:* `annotate_availability` is applied when preparing data, but `TradingEnv.step` executes trades for every index row without inspecting availability flags or market hours, treating all timesteps as tradeable.【F:data/data_handler.py†L20-L45】【F:environment/trading_env.py†L123-L185】

- **Deterministic execution ordering and RNG control for reproducibility**
  *Why:* Trade execution order is randomized via `random.shuffle` with no seeded RNG, which means identical inputs can yield different fills and portfolio trajectories. Introducing seeded ordering (or deterministic priority rules) and persisting the seed with run manifests would make training/backtests replayable and trustworthy.
  *Evidence:* `execute_all_trades` shuffles buy-order pairs before processing, and no seed is set or recorded anywhere in the environment lifecycle.【F:core/portfolio_manager.py†L248-L293】

- **Asset-specific compliance guards (shortability, lot sizes, borrow costs)**
  *Why:* Order sizing ignores borrow constraints, lot sizes, or per-asset prohibitions, so the agent can short assets that may be unshortable or size orders in fractional lots. Adding symbol-level compliance rules and borrow-cost accounting would convert policy outputs into broker-legal orders and prevent unrealistic positions.
  *Evidence:* `_calculate_trade_quantity` derives size solely from balance/position and a global percentage, while `_handle_sell` updates positions without checking borrow availability or rounding to permitted lots.【F:core/portfolio_manager.py†L240-L295】【F:core/portfolio_manager.py†L33-L75】

_Consolidated after accepting all reviewer suggestions; treat this as the canonical impact roadmap._
