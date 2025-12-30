# Section B: Architectural Decomposition Plan

This document captures a stepwise decomposition for the monolithic trading agent codebase. It assigns every major class/function to a cohesive module boundary and details interaction rules so teams can own subsystems independently without loss of functionality.

## Goals
- Keep `UnifiedTradingAgent` as the execution orchestrator while delegating specialized responsibilities to focused modules.
- Centralize preprocessing and availability-aware validation in the data layer to avoid duplicated logic in environments.
- Make feature pipelines injectable from PPO/agent configuration instead of being created implicitly inside the environment.
- Preserve reusable evaluation and metrics streaming via modular callbacks.

## Modules and Assignments

### 1) Agent Orchestration
- **Primary class**: `UnifiedTradingAgent` in `src/core/base_agent.py` orchestrates environment initialization, PPO setup, training/test flows, checkpointing, and evaluation callback wiring.【F:src/core/base_agent.py†L42-L357】
- **Key responsibilities**: `_init_state`, `_build_training_schedule`, `_get_data_index`, `initialize_env`, `_init_env_and_model`, `_create_validation_env`, `_prepare_evaluation_config`, `train`, `test`, `predict`, `save/load` methods compose dependent services without owning their logic.【F:src/core/base_agent.py†L58-L247】【F:src/core/base_agent.py†L247-L516】
- **Exports**: Agent façade for UI and automation layers; dependency injection points for feature pipelines (`feature_pipeline` argument) and evaluation callbacks.

### 2) Environment & Rewards
- **Classes/functions**: `TradingEnv` and `fetch_trading_data` in `src/environment/trading_env.py` manage Gym environment state, observation construction, and reward calculator wiring.【F:src/environment/trading_env.py†L24-L153】【F:src/environment/trading_env.py†L154-L266】
- **Responsibilities**: Consume preprocessed OHLCV frames from the data layer, apply optional supplied feature pipelines, and expose `step`/`reset` semantics. Reward shaping is delegated to `RewardsCalculator`.
- **Boundary rule**: No direct provider/DB calls; relies on data manager output via `fetch_trading_data`.

### 3) Data Management & Validation
- **Classes/functions**: `TradingDataManager.fetch`, availability-aware validation helpers (`ensure_utc_timestamp`, `validate_ohlcv_frame`, `annotate_availability`) in `src/data/data_handler.py` and `src/data/validation` package.【F:src/data/data_handler.py†L1-L53】
- **Responsibilities**: Validate OHLCV schema, align close-of-day fields, annotate availability times, and serve preprocessed frames to environments and agents. Provider implementations (e.g., `LocalCacheProvider`, `LiveAPIProvider`, `FileSystemProvider`) enforce the same validation contract.【F:src/data/data_handler.py†L22-L53】【F:src/data/providers.py†L1-L57】
- **Exports**: Provider protocol plus data manager API returning ready-to-use frames.

### 4) Feature Engineering & Prediction
- **Classes/functions**: Feature sources, registry, execution engine, cache, selectors, and prediction models in `src/data/feature_engineering` (not enumerated here). Environment accepts an injected `feature_pipeline` instead of constructing processors internally.【F:src/environment/trading_env.py†L44-L102】
- **Responsibilities**: Offer configurable pipelines for PPO data sourcing; maintain registries and caches; compute model predictions.

### 5) Metrics & Portfolio
- **Classes/functions**: `PortfolioManager` metrics accessors, `MetricsCalculator` analytics, and `MetricStreamingEvalCallback` used for periodic evaluation during training.【F:src/core/base_agent.py†L282-L357】【F:src/environment/trading_env.py†L1-L70】
- **Responsibilities**: Compute portfolio performance, stream metrics via callbacks, and provide reusable visualization hooks.

### 6) Hyperparameter Tuning & Experiments
- **Classes/functions**: Experiment registry and hyperparameter optimization utilities in `src/core/experiments.py` and `src/core/hyperparameter_search.py` (not enumerated here).
- **Responsibilities**: Register experiment manifests, run tuning workflows, and store artifacts referenced by the agent.

### 7) Broker & Execution
- **Classes/functions**: Broker adapters and configuration loaders in `src/core/brokers/*` plus IBKR integration scripts; these provide execution endpoints and DTOs for orders and accounts.
- **Responsibilities**: Abstract broker APIs; prevent external types from leaking into agent/environment layers.

### 8) UI & Visualization
- **Classes/functions**: Streamlit tabs in `app/components/*` and chart helpers in `src/core/visualization.py` form the presentation layer. They consume agent and service facades without direct provider/DB access.

### 9) Utilities & Config
- **Classes/functions**: Common validators, logging utilities, data splitters, and configuration constants in `src/utils/*` and `src/core/config.py`. These are shared helpers with no domain logic.

## Interaction Flow (Textual Dependency Diagram)
UI → Agent Orchestration → Environment → Data Management & Validation → Feature Engineering → Metrics/Portfolio → Brokers (execution). Cross-cutting: Hyperparameter Tuning & Experiments feed into Agent configuration; Utilities & Config support all modules.

## Boundary Rules
- Environment must only consume prevalidated, availability-aware data frames from Data Management; it must not call providers or DB APIs directly.【F:src/environment/trading_env.py†L24-L79】
- Feature pipelines are injected via agent configuration (`feature_pipeline`), keeping feature selection out of the environment internals.【F:src/environment/trading_env.py†L44-L102】
- UI components call the Agent façade and must not bypass service layers to reach data providers or brokers.
- Metrics/portfolio computations remain reusable and free of UI or provider dependencies.

## Rationale
This decomposition assigns cohesive responsibilities, minimizes cross-module dependencies, and preserves all existing functions by relocating them into explicit domain modules. It allows teams (RL, Data Platform, Feature/ML, Quant Analytics, MLOps, Brokers, Frontend) to own their modules independently while maintaining stable typed interfaces across boundaries.
