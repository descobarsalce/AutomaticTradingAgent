
# Advanced Algorithmic Trading Platform

Comprehensive reinforcement-learning trading workstation with a Streamlit UI, availability-aware data layer, modular feature engineering, and deterministic checkpointing for reproducible research.

## Table of Contents
- [How to use this README](#how-to-use-this-readme)
- [Documentation map (start here)](#documentation-map-start-here)
- [Highlights](#highlights)
- [Architecture at a glance](#architecture-at-a-glance)
- [Repository layout](#repository-layout)
- [Requirements & environment setup](#requirements--environment-setup)
- [Data sources, caching, and credentials](#data-sources-caching-and-credentials)
- [Running the Streamlit app](#running-the-streamlit-app)
- [Programmatic usage](#programmatic-usage)
- [Training, tuning, and checkpoints](#training-tuning-and-checkpoints)
- [Feature engineering pipeline](#feature-engineering-pipeline)
- [Testing](#testing)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)

## How to use this README
This README is the **single entry point** for onboarding. It explains the codebase at a high level and points to the precise documents that contain deeper instructions. If you are a new agent, start with the **Documentation map** below and follow the section that matches your task (feature engineering, database setup, broker integration, architecture, or dependencies).

## Documentation maintenance (required for agents)
When you make changes that affect behavior, configuration, dependencies, or workflow, you **must** update the relevant documentation in the same change. This keeps the README and the linked docs accurate and prevents drift.

**Update docs when you change:**
- **Dependencies or setup steps** → `docs/dependencies.md` and the setup section in this README.
- **Feature engineering pipeline or API** → `docs/FeatureEngineerImprovementPlan.md` (roadmap/architecture) and the Feature Engineering section in this README if behavior changed.
- **Database setup or storage behavior** → `docs/local_db_setup.md` and the Data sources/caching section in this README.
- **Broker/Lean/IBKR workflows** → `docs/quantconnect_ibkr_overview.md`.
- **Architecture/module boundaries** → `docs/section_b_architecture.md` and the Architecture/Repository layout sections here.

## Documentation map (start here)
Use this map to jump to the right instruction set quickly. Each document is authoritative for its topic.

| Task / Topic | Go to | What you'll find there |
| --- | --- | --- |
| Feature engineering roadmap & architecture | `docs/FeatureEngineerImprovementPlan.md` | Full migration plan, plugin architecture, selection/competition, data sources, caching, and phased roadmap. |
| Local database setup (SQLite/Postgres) | `docs/local_db_setup.md` | Mac-focused local DB instructions, `DATABASE_URL` guidance, and connectivity checks. |
| Lean + IBKR integration | `docs/quantconnect_ibkr_overview.md` | Roles of Lean/IBKR, paper-first workflow, setup plan, and example commands. |
| Architectural decomposition plan | `docs/section_b_architecture.md` | Module boundaries, responsibilities, and cross-module interaction rules. |
| Dependency manifest & install options | `docs/dependencies.md` | Version list and full dependency install command (mirrors `pyproject.toml`). |

## Highlights
- **Unified PPO agent** orchestrates environment creation, training, evaluation callbacks, manifests, and checkpoint integrity checks (`src/core/base_agent.py`).
- **Streamlit workstation** with tabs for data exploration, feature selection, ML feature models, hyperparameter tuning, training, testing, and technical analysis (`app/main.py`, `app/components/`).
- **Data layer with availability annotations**: SQL-backed cache plus live fetchers that normalize OHLCV schema, enforce UTC windows, and shift prior-day fields for next-session alignment (`src/data/data_handler.py`, `src/data/providers.py`).
- **Feature pipeline injection**: optional `FeatureProcessor` or precomputed feature DataFrame passed directly into the environment; predictions (e.g., LSTM horizons) and normalization are supported (`src/data/feature_engineering/feature_processor.py`).
- **Risk-aware environment**: blended weight actions with gating, position sizing caps, transaction costs, and risk budget metrics surfaced alongside reward shaping (`src/environment/trading_env.py`).
- **Experiment hygiene**: manifests hash model state, log PPO/env params, and gate inference to deterministic settings when requested. Intermediate checkpoints and evaluation streams are captured alongside TensorBoard logs (`src/core/base_agent.py`, `src/core/experiments.py`).
- **Broker integrations ready**: Lean/IBKR adapters and scripts live under `integrations/` and `scripts/` to make paper/live connectivity possible once credentials are provided.

## Architecture at a glance
- **UI**: Streamlit tabs coordinate data fetches, feature selection, tuning, and agent controls (`app/main.py`, `app/components/`).
- **Agent orchestration**: `UnifiedTradingAgent` wires environments, PPO, callbacks, manifests, deterministic seeds, and checkpoint I/O (`src/core/base_agent.py`, `src/core/checkpoint_io.py`).
- **Environment**: `TradingEnv` consumes prevalidated OHLCV frames, applies optional feature pipelines, constrains weights, charges transaction costs, and computes shaped rewards (`src/environment/trading_env.py`).
- **Data layer**: `DataHandler` + providers fetch/caches OHLCV, validate schema, align prior-day fields, and annotate availability (`src/data/data_handler.py`, `src/data/providers.py`, `src/data/validation/*`).
- **Features & predictions**: `FeatureProcessor` can compute engineered features and optional LSTM predictions, normalize them, and expose observation vectors (`src/data/feature_engineering/*`).
- **Hyperparameter search**: `IterativeHyperparameterOptimizer` runs exploration→exploitation Optuna cycles with schedule-aware progress (`src/core/hyperparameter_search.py`).
- **Metrics & viz**: `src/metrics/` holds calculators and streaming sinks; `src/core/visualization.py` produces action/price charts consumed in UI and callbacks.

## Repository layout
```
├── app/                       # Streamlit entrypoint + tab implementations
├── src/                       # Core library code (agent, data, env, metrics, utils)
├── config/                    # Configuration and example configs
├── scripts/                   # Utility scripts (e.g., IBKR session runner)
├── archive/                   # Legacy/unused files retained for reference
├── docs/                      # Additional documentation (architecture, broker integration, local DB)
└── tests/                     # Pytest suite
```

### Module reference by directory
- **UI (`app/main.py`, `app/components/`)**
  - `database_tab.py`, `features_tab.py`, `ml_models_tab.py`, `tuning_tab.py`, `training_tab.py`, `testing_tab.py`, `analysis_tab.py`, and `execution_window_ui.py` render the Streamlit workflows and persist state into `st.session_state`.
- **Agent & training (`src/core/`)**
  - `base_agent.py` coordinates env setup, PPO wiring, training/test loops, manifests, and deterministic seeds.
  - `env_factory.py` builds environments; `schedule_builder.py` derives rollout lengths; `callbacks/` hosts eval/streaming callbacks; `experiments.py` and `checkpoint_io.py` handle manifests/checkpoints.
  - `hyperparameter_search.py` runs Optuna exploration→exploitation cycles; `training_functions.py` bridges Streamlit controls to the agent; `testing_functions.py` exposes inference helpers.
  - Portfolio, risk, and token modeling utilities live in `portfolio_manager.py`, `risk_budget_manager.py`, and `market_token_model.py`; token schemas are defined in `token_schema.py`.
  - Broker adapters and config loaders reside in `src/core/brokers/` (IBKR-focused).
- **Environment (`src/environment/`)**
  - `trading_env.py` implements the Gymnasium environment, reward shaping, risk controls, action gating, and observation assembly; `rewards_calculator.py` encapsulates reward logic.
- **Data (`src/data/`)**
  - `data_handler.py` + `providers.py` fetch/cache OHLCV data; `database.py` defines SQLAlchemy tables; `data_SQL_interaction.py` manages sessions; `validation/` enforces OHLCV/time rules (availability, timezone, schema).
  - `stock_downloader.py` unifies Yahoo and Alpha Vantage (equities + options); `data_feature_engineer.py` is the legacy feature pipeline (deprecated).
  - `feature_engineering/` holds the modular pipeline (`feature_processor.py`, registry/sources/selectors/predictions).
- **Metrics (`src/metrics/`)**
  - `metrics_calculator.py` and `base_indicators.py` compute portfolio/indicator stats; `metric_sink.py` streams metrics for callbacks and UI plots.
- **Utilities (`src/utils/`)**
  - Common constants (`common.py`), data splits (`data_splitter.py`), progress callbacks (`callbacks.py`), DB config/pooling (`db_config.py`), logging helpers (`logging_utils.py`), and stock parsing (`stock_utils.py`).
- **Scripts, configs, and plans**
  - `scripts/run_ibkr_session.py` smoke-tests IBKR connectivity using `config/ibkr.paper.example.yaml`.
  - `docs/FeatureEngineerImprovementPlan.md` outlines the roadmap for the feature pipeline; `docs/dependencies.md` lists environment notes.
  - `debug_training.py` offers a synthetic-data harness for debugging LSTM prediction features.

## Requirements & environment setup
- Python **3.11+**.
- Recommended installer: [uv](https://docs.astral.sh/uv/) (a fast `pip`/`pip-tools` replacement) because the repo ships an `uv.lock`.

Setup steps:
```bash
python -m pip install -U uv           # once, to get uv
uv venv .venv                         # create a virtual environment
source .venv/bin/activate             # or .venv\\Scripts\\activate on Windows
uv pip sync uv.lock                   # install all pinned dependencies
```

If you prefer classic tooling, you can also install from `pyproject.toml` with `pip install -e .` after creating a virtualenv, but `uv pip sync` is the reproducible path the project is maintained against.

## Data sources, caching, and credentials
- **Providers**: use `LocalCacheProvider` (SQLite-first), `LiveAPIProvider` (always hit APIs), `SessionStateProvider` (uses the Streamlit session’s `DataHandler`), or `FileSystemProvider` for CSV/Parquet (`src/data/providers.py`).
- **Downloads**: `StockDownloader` pulls equities from Yahoo or Alpha Vantage and Yahoo option chains in a unified schema (`src/data/stock_downloader.py`).
- **Caching**: `DataHandler` writes normalized OHLCV frames into a SQL store; SQLite (`trading_data.db`) is the default unless `DATABASE_URL` is set. Connection pooling and health checks are configured in `src/utils/db_config.py`.
- **Alpha Vantage key**: place `ALPHA_VANTAGE_API_KEY` in `.streamlit/secrets.toml` when using that source.
- **Timezone & validation**: all fetches enforce UTC windows, rename/validate OHLCV columns, and shift prior-day fields to align next-session observations (`src/data/data_handler.py`, `src/data/validation`).

## Running the Streamlit app
```bash
streamlit run app/main.py --server.address 0.0.0.0 --server.port 8501
```
- Open `http://localhost:8501`.
- If port `8501` is occupied, change `--server.port` (e.g., `8502`).
- The app defaults to the local SQLite cache; configure `DATABASE_URL` for Postgres (see `docs/onboarding/local_db_setup.md` for guidance).

### What the UI offers
- **Database Explorer**: query cached OHLCV data and inspect coverage.
- **Feature Selection & ML Feature Models**: configure feature sources, normalization, and prediction models that feed the environment.
- **Hyperparameter Tuning**: Optuna-driven iterative exploration→exploitation cycles with progress bars (`src/core/hyperparameter_search.py`).
- **Model Training & Testing**: launch training with optimized or manual PPO params, monitor metrics, and visualize trades (`app/components/training_tab.py`, `app/components/testing_tab.py`).
- **Technical Analysis**: chart overlays and indicator exploration for loaded symbols.
- **TensorBoard**: training runs log under `artifacts/logs/tensorboard/` by default; launch with `tensorboard --logdir artifacts/logs/tensorboard`.

## Programmatic usage
The agent façade requires an explicit data provider and accepts an optional feature pipeline.

```python
from datetime import datetime
from src.core.base_agent import UnifiedTradingAgent
from src.data.providers import LocalCacheProvider
from src.data.feature_engineering.feature_processor import FeatureProcessor

provider = LocalCacheProvider()  # uses SQLite cache, falls back to downloads
feature_proc = FeatureProcessor(
    feature_config={
        "use_feature_engineering": True,
        "normalize_features": True,
        "sources": {"technical": {"enabled": True, "features": ["rsi", "ema_20"]}},
    },
    symbols=["AAPL", "MSFT"],
)

agent = UnifiedTradingAgent()
metrics = agent.train(
    stock_names=["AAPL", "MSFT"],
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 6, 1),
    env_params={"initial_balance": 10000, "transaction_cost": 0.001, "max_pct_position_by_asset": 0.2},
    ppo_params={"n_steps": 512, "batch_size": 128},
    feature_pipeline=feature_proc,      # FeatureProcessor or a precomputed DataFrame
    checkpoint_interval=5000,
    provider=provider,
)

# Deterministic inference with manifest hash checks
agent.initialize_env(
    stock_names=["AAPL", "MSFT"],
    start_date=datetime(2023, 6, 2),
    end_date=datetime(2023, 7, 1),
    env_params={"initial_balance": 10000, "transaction_cost": 0.001},
    feature_pipeline=feature_proc,
    training_mode=False,
    provider=provider,
)
agent._set_deterministic_seed(1234)
agent.load(
    path="artifacts/checkpoints/<run_id>/trained_model.zip",
    manifest_path="artifacts/checkpoints/<run_id>/manifest.json",
    deterministic_eval=True,
)
obs, _ = agent.env.reset(seed=1234)
action = agent.predict(obs)  # deterministic when the manifest hash matches
```

## Training, tuning, and checkpoints
- **Schedules & evaluation**: `build_training_schedule` computes timesteps from available data; `MetricStreamingEvalCallback` runs deterministic validation rollouts on fixed seeds and streams metrics to sinks/plots (`src/core/base_agent.py`, `src/core/schedule_builder.py`, `src/core/callbacks.py`).
- **Hyperparameter optimization**: `IterativeHyperparameterOptimizer` alternates exploration/exploitation rounds and refines PPO ranges until improvements stall (`src/core/hyperparameter_search.py`). The Streamlit tuning tab is the easiest entrypoint.
- **Manifests**: every training run writes `artifacts/checkpoints/<run_id>/manifest.json` recording PPO/env params, feature config, data range, deterministic-eval flag, and a hash of the model state dict. `load` and `load_for_inference` verify hashes before allowing predictions.
- **Checkpoint cadence**: pass `checkpoint_interval` to `UnifiedTradingAgent.train` to emit intermediate PPO checkpoints under `artifacts/checkpoints/<run_id>/checkpoints/` alongside TensorBoard logs.
- **Evaluation streams**: `MetricStreamingEvalCallback` can log JSONL metrics via `MetricsSink` while also rendering price/action charts for inspection in `artifacts/reports/<run_id>/eval_stream/`.
- **Manual vs optimized params**: the Training tab lets you toggle manual PPO parameters or reuse Optuna results saved in session state (`app/components/training_tab.py`).
- **Testing loop**: the Testing tab reuses the configured agent to run inference with deterministic seeds and displays trade-by-trade metrics; programmatically call `agent.test(...)` after loading a checkpoint.

## Feature engineering pipeline
- Enable via `feature_pipeline` when initializing or training the agent. Supported inputs:
  - `FeatureProcessor` instance (auto-initializes, normalizes, and can append prediction columns).
  - `pandas.DataFrame` with the same index as trading data.
  - `dict` or `list` of column names to select from raw data.
- Observation vectors can include engineered features, prediction horizons, raw OHLCV, positions, and balance. Observation dimensions adjust automatically based on enabled components (`src/environment/trading_env.py`, `src/data/feature_engineering/feature_processor.py`).
- When `use_predictions` is enabled, the prediction engine can train LSTM forecasters for configured horizons and append normalized outputs to the observation space.
- Set `include_raw_prices`, `include_positions`, and `include_balance` flags in `feature_config` to trim or expand the observation footprint for custom experiments.

## Testing
Run the test suite from the repo root:
```bash
python -m pytest tests/ -v
```
- Streamlit UI smoke tests are not automated; run the app locally to validate tab interactions when changing UI components.

## Additional docs
- **Architecture decomposition**: `docs/section_b_architecture.md`
- **Broker + Lean integration**: `docs/quantconnect_ibkr_overview.md`
- **Local database setup**: `docs/local_db_setup.md`
- **Feature engineering roadmap**: `docs/FeatureEngineerImprovementPlan.md`

## Limitations
- Training can be compute-intensive; default config forces CPU-only Torch to avoid GPU portability issues.
- Live market data is subject to provider latency/quotas; Alpha Vantage requires an API key and rate-limiting can slow backfills.
- Real execution is not wired by default; IBKR/Lean adapters are available but must be configured before any live deployment.
- Optuna/pruning workloads can be heavy; consider reducing trial counts or disabling pruning when running on resource-constrained machines.

## Troubleshooting
- **Empty Yahoo downloads**: ensure date ranges are valid and the ticker exists; the downloader logs direct API fallback attempts in `src/data/stock_downloader.py`.
- **Alpha Vantage errors**: confirm `ALPHA_VANTAGE_API_KEY` is set in `.streamlit/secrets.toml`; API rate limits may require retries or a cached provider.
- **Database locks (SQLite)**: avoid concurrent writers; if you need multi-user access, point `DATABASE_URL` to Postgres and restart the app.
- **Manifest hash mismatch**: re-load the checkpoint with the matching `manifest.json` to enforce determinism; mismatches indicate an altered model file.
- **Missing feature columns**: when passing custom DataFrames as `feature_pipeline`, ensure indices align with the trading data; the environment will reindex but may fill gaps with NaNs.
