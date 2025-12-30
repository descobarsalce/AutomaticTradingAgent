# Repository Reorganization Plan (Phases 1–5)

This document captures the evidence-first reorganization plan and outputs for phases 1–5. It is the authoritative, file-based record of the audit, target architecture, migration map, refactor checklist, and documentation guardrails.

## Phase 1 — Repository Audit (Diagnosis Only)

> **Note:** The snapshot below reflects the current layout after the initial folder migrations (app/src/artifacts). The Phase 3 migration map preserves the original pre-migration paths for traceability.

### 1.1 Collapsed directory tree (depth ≤ 2)
```
.
├── archive
├── assets
│   └── screenshots
├── app
│   └── components
├── config
├── src
│   ├── core
│   │   ├── brokers
│   │   └── callbacks
│   ├── data
│   │   ├── feature_engineering
│   │   └── validation
│   ├── environment
│   ├── metrics
│   └── utils
├── docs
│   ├── architecture
│   ├── onboarding
│   ├── reference
│   ├── roadmap
│   └── planning
├── integrations
│   └── lean_ibkr
├── scripts
├── tests
│   ├── feature_engineering
│   ├── integration
│   └── unit
├── artifacts
│   └── models
│       ├── LSTM_AAPL
│       └── Transformer_ABNB
├── .streamlit
├── .venv
└── .git
```

### 1.2 File classification (depth ≤ 2, evidence-based)
| File/Path | Classification |
| --- | --- |
| `README.md` | Documentation |
| `docs/*` | Documentation |
| `app/main.py` | Orchestration / entrypoint |
| `scripts/debug_training.py` | Experiment / exploration |
| `scripts/run_ibkr_session.py` | Pipeline / orchestration |
| `app/components/*` | UI / application layer |
| `src/core/*` | Core library code |
| `src/data/*` | Core library code (data layer) |
| `src/environment/*` | Core library code (environment) |
| `src/metrics/*` | Core library code (metrics) |
| `src/utils/*` | Utilities / helpers |
| `config/*` | Configuration |
| `tests/*` | Tests |
| `artifacts/models/*` | Outputs (models) |
| `archive/*` | Archived artifacts |
| `.streamlit/*` | Configuration (app) |
| `pyproject.toml`, `uv.lock`, `pytest.ini` | Configuration / tooling |
| `.git/*`, `.venv/*`, `.DS_Store` | Tooling / environment metadata |

### 1.3 Evidence-based findings (diagnosis only)
- **Notebooks used as production code:** none detected at depth ≤ 2 (no `.ipynb` files).
- **Duplicate logic, scripts mixing concerns, hidden coupling, naming inconsistencies:** require content-level inspection beyond filename inventory. These are flagged for inspection in Phase 4 refactors.

## Phase 2 — Canonical Target Architecture

### 2.1 Proposed directory tree
```
.
├── README.md
├── docs/
│   ├── README.md
│   ├── architecture/
│   │   └── section_b_architecture.md
│   ├── onboarding/
│   │   ├── local_db_setup.md
│   │   └── quantconnect_ibkr_overview.md
│   ├── roadmap/
│   │   └── feature_engineering_plan.md
│   ├── reference/
│   │   └── dependencies.md
│   └── planning/
│       └── architecture_reorg_plan.md
├── src/
│   ├── core/
│   ├── data/
│   ├── environment/
│   ├── metrics/
│   └── utils/
├── app/
│   ├── main.py
│   └── components/
├── scripts/
├── config/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── feature_engineering/
├── artifacts/
│   ├── models/
│   ├── reports/
│   └── logs/
├── archive/
├── integrations/
├── assets/
│   └── screenshots/
├── .streamlit/
├── pyproject.toml
├── uv.lock
├── pytest.ini
└── .gitignore
```

### 2.2 Top-level responsibilities
- `docs/`: authoritative documentation grouped by purpose.
- `src/`: production library code.
- `app/`: UI entrypoint and UI components.
- `scripts/`: orchestration utilities and CLI runners.
- `config/`: configuration files and examples.
- `tests/`: tests grouped by type.
- `artifacts/`: generated outputs (models/logs/reports).
- `archive/`: retained legacy artifacts.
- `integrations/`: external platform integrations.
- `assets/`: static assets for UI/docs.
- `.streamlit/`: Streamlit configuration.

### 2.3 Justifications
- `app/` vs `src/`: keeps UI separate from reusable library code.
- `artifacts/`: formalizes output locations for reproducibility.
- `docs/` subfolders: separates roadmap vs onboarding vs reference for clarity.

## Phase 3 — Deterministic Migration Map (Design-Only)

| Old Path | New Path | Action | Rationale |
| --- | --- | --- | --- |
| `docs/architecture/section_b_architecture.md` | `docs/architecture/section_b_architecture.md` | MOVE (keep) | Architecture doc location fits target tree. |
| `docs/onboarding/local_db_setup.md` | `docs/onboarding/local_db_setup.md` | MOVE (keep) | Onboarding doc location fits target tree. |
| `docs/onboarding/quantconnect_ibkr_overview.md` | `docs/onboarding/quantconnect_ibkr_overview.md` | MOVE (keep) | Onboarding doc location fits target tree. |
| `docs/roadmap/feature_engineering_plan.md` | `docs/roadmap/feature_engineering_plan.md` | MOVE (keep) | Roadmap doc location fits target tree. |
| `docs/reference/dependencies.md` | `docs/reference/dependencies.md` | MOVE (keep) | Reference doc location fits target tree. |
| `main.py` | `app/main.py` | MOVE | Separate UI entrypoint from library code. |
| `components/*` | `app/components/*` | MOVE | Co-locate UI components with app entry. |
| `core/*` | `src/core/*` | MOVE | Core library code in `src/`. |
| `data/*` | `src/data/*` | MOVE | Data layer in `src/`. |
| `environment/*` | `src/environment/*` | MOVE | Environment logic in `src/`. |
| `metrics/*` | `src/metrics/*` | MOVE | Metrics logic in `src/`. |
| `utils/*` | `src/utils/*` | MOVE | Utilities in `src/`. |
| `scripts/*` | `scripts/*` | MOVE (keep) | Orchestration stays in scripts. |
| `.prediction_models/*` | `artifacts/models/*` | MOVE | Formalize outputs under artifacts. |
| `debug_training.py` | `scripts/debug_training.py` | MOVE | Treat as a runnable script. |
| `tests/*` | `tests/{unit,integration,feature_engineering}/*` | MOVE | Group tests by type. |
| `config/*` | `config/*` | MOVE (keep) | Config remains top-level. |
| `integrations/*` | `integrations/*` | MOVE (keep) | Integrations remain separate. |
| `assets/*` | `assets/*` | MOVE (keep) | Assets remain top-level. |

## Phase 4 — Required Refactors (Post-Move)

1) **Scripts → library extraction**
   - `scripts/debug_training.py`: move reusable logic into `src/` modules; keep script thin.
   - `scripts/run_ibkr_session.py`: ensure core logic lives in `src/core/brokers/` or `integrations/`.

2) **Config externalization**
   - Review `app/main.py`, `src/data/data_handler.py`, `src/utils/db_config.py`, `scripts/run_ibkr_session.py` for hardcoded paths or credentials and replace with config/env.

3) **API formalization**
   - Feature pipeline: `src/data/feature_engineering/*` ↔ `src/environment/trading_env.py` should use a stable, documented interface.
   - Broker adapters: `src/core/brokers/*` should define a clear boundary to prevent leakage into agent/environment layers.

4) **Test regrouping**
   - Move tests into `tests/unit`, `tests/integration`, `tests/feature_engineering` based on scope.

5) **Artifact paths**
   - Ensure all prediction model references point to `artifacts/models/*`.

## Phase 5 — Documentation & Guardrails

### 5.1 Root README outline
1. Project summary
2. Documentation map
3. Quickstart
4. Architecture overview
5. Repo layout
6. Contribution rules (doc updates required)
7. Testing & CI
8. Troubleshooting
9. Roadmap pointer

### 5.2 Folder README templates
- `docs/README.md`: documentation index.
- `src/README.md`: library boundaries and API contracts.
- `app/README.md`: UI architecture and Streamlit state conventions.
- `scripts/README.md`: runnable scripts list and usage.
- `tests/README.md`: unit vs integration vs feature engineering.

### 5.3 Naming conventions
- Files: `snake_case.py`
- Tests: `test_*.py`
- Streamlit tabs: `*_tab.py`
- Experiments: `<date>_<model>_<dataset>_<seed>`
- Configs: `<system>.<env>.yaml`
- Outputs: `artifacts/models/<run_id>/`, `artifacts/logs/<run_id>/`, `artifacts/reports/<run_id>/`

### 5.4 Guardrails
1) No new top-level folders without README update.
2) Scripts must orchestrate only (no duplicated core logic).
3) Outputs must go under `artifacts/`.
4) Roadmap docs must be labeled as future work.
