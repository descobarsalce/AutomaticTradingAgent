# Performance Improvement Plan (Comprehensive)

This plan focuses on improving financial performance through data quality, signal robustness, execution realism, and risk management. It is designed to be executed iteratively with measurable checkpoints.

## MVP Focus (high-impact, low-effort)
For a small-capital showcase, prioritize these first:

1) **Data integrity & leakage checks** to avoid inflated backtests.
2) **Risk-aware reward shaping** (small drawdown/volatility/turnover penalties).
3) **Basic transaction cost model** (flat bps assumption).
4) **Single benchmark comparison** (SPY buy-and-hold or equal-weight).
5) **Hard risk caps** (max position size, max daily loss).

Only expand to later phases once the MVP signals are stable and reproducible.

## Success Criteria (to track every phase)
- Improve risk-adjusted performance (Sharpe/Sortino) vs baseline by a predefined threshold.
- Maintain or reduce maximum drawdown while improving return.
- Demonstrate stability across at least two out-of-sample regimes.
- Keep turnover within predefined limits to avoid unrealistic execution.

## Phase 1 — Baseline & Diagnostics
**Goal:** Establish current performance, identify failure modes, and lock reproducibility.

- Run a fixed, reproducible backtest (symbols, dates, seeds, env params).
- Record baseline metrics: Sharpe, Sortino, CAGR, max drawdown, volatility, turnover, hit rate.
- Capture regime diagnostics: worst drawdowns by date range, per‑symbol contribution, trade distribution.
- Save artifacts (metrics + plots) under `artifacts/reports/<run_id>/`.
- Add baseline comparison against at least one simple benchmark (e.g., buy‑and‑hold).

## Phase 2 — Data & Feature Quality
**Goal:** Improve signal quality and reduce leakage or noise.

- Validate OHLCV completeness, time alignment, and missing data handling.
- Audit engineered features for leakage and regime instability.
- Perform feature ablations (remove/keep groups) with fixed seeds.
- Expand feature set only when ablations show consistent improvement.
- Track feature drift metrics across regimes.

## Phase 3 — Reward & Risk Shaping
**Goal:** Align learning objective with risk‑adjusted performance.

- Review reward function and remove incentives for churn.
- Add drawdown/volatility penalties or risk‑adjusted rewards.
- Re‑evaluate under the same baseline setup to isolate changes.
- Validate reward stability across different volatility regimes.

## Phase 4 — Execution Realism
**Goal:** Reduce backtest/live drift by modeling friction.

- Confirm transaction costs + slippage are realistic by asset class.
- Add spread or liquidity proxies where applicable.
- Stress test against higher friction to evaluate robustness.
- Document execution assumptions clearly for auditability.

## Phase 5 — Training Strategy & Validation
**Goal:** Improve generalization, avoid overfitting.

- Constrain PPO search space and use walk‑forward validation.
- Evaluate models on out‑of‑sample windows.
- Choose models by risk‑adjusted returns, not raw PnL.
- Record parameter sensitivity for top candidates.

## Phase 6 — Portfolio Construction & Risk Controls
**Goal:** Stabilize outputs and reduce tail risk.

- Apply position sizing caps, exposure limits, and max loss thresholds.
- Consider volatility targeting or risk parity weighting.
- Validate diversification and exposure concentration.
- Enforce portfolio constraints at both training and inference time.

## Phase 7 — Monitoring & Iteration
**Goal:** Track drift and maintain performance.

- Log live/paper performance with the same metrics as backtests.
- Compare performance drift and schedule retrains based on thresholds.
- Only promote changes after statistical validation.
- Maintain a changelog of model versions and performance deltas.

---

# Questions for Finance Agent A (Signals & Strategy)

1) Which **signal families** are most robust for short‑ to medium‑horizon equities (momentum, mean‑reversion, carry, volatility, macro)?
2) Which **feature transformations** tend to improve RL stability (e.g., log returns, volatility scaling, regime indicators)?
3) What are common **leakage pitfalls** in OHLCV‑based features?
4) What **baselines/benchmarks** should we compare against (SPY buy‑and‑hold, equal‑weight, volatility‑targeted)?
5) For RL agents, do you prefer **Sharpe vs Sortino vs Calmar** as the primary target metric?
6) Which **regime filters** are most useful (volatility regime, trend regime, macro regime)?
7) Are there **cross‑asset signals** worth adding (SPY/QQQ correlations, sector spreads)?
8) How should we handle **non‑stationarity** (rolling normalization windows, regime detection, adaptive features)?
9) What **signal validation protocol** do you recommend (IC testing, rank IC, walk‑forward)?
10) What is the minimum **signal half‑life** you’d target for this model horizon?
11) How should we treat **overlapping signals** (feature pruning, correlation caps)?
12) Which **feature families** are low‑risk but high‑value to add first?
13) When is **feature selection** (e.g., SHAP, mutual info) actually harmful?
14) Do you recommend **ensemble signals** or a single robust pipeline?
15) Which **asset universe** is best for early optimization (liquid large‑caps, ETFs)?
16) What **risk factors** (size, value, momentum, quality) should we explicitly capture?
17) Are there **seasonality effects** worth incorporating?
18) What **macro indicators** are worth including for regime detection?
19) Which **data sources** have the best cost/benefit tradeoff for signal quality?
20) How should we **validate signal decay** over time?

---

# Questions for Finance Agent B (Execution, Risk & Portfolio)

1) What is a realistic **transaction cost + slippage model** for U.S. equities? For options?
2) What is a reasonable **spread or liquidity proxy** if we lack order book data?
3) Which **risk constraints** matter most (max drawdown, max daily loss, max leverage, exposure caps)?
4) For RL outputs, do you recommend **weight clipping** or **volatility targeting**?
5) How should we **stress test** strategies against higher costs or volatility regimes?
6) How should we model **partial fills** or order execution delays?
7) What is the best **portfolio construction approach** to stabilize RL (risk parity, minimum variance, equal risk contribution)?
8) How do you handle **correlated positions** (sector caps, correlation‑based limits)?
9) What **turnover thresholds** are acceptable for this strategy?
10) How should we model **borrow costs** or shorting constraints?
11) What **live vs backtest drift checks** do you trust most?
12) What are the best practices for **retraining cadence**?
13) Which **risk‑adjusted metric** should control production promotion?
14) What are the minimum **liquidity filters** you’d apply to the universe?
15) How would you define **kill‑switch criteria** for live trading?
16) What **stress scenarios** should we simulate (flash crash, rate shock, liquidity freeze)?
17) How should we set **risk budgets** by asset class?
18) What **order types** are safest to assume in backtests?
19) What is a reasonable **capacity estimate** for this strategy?
20) How should we monitor **transaction cost drift** over time?

---

# Minimal MVP Questions (use these if time/budget is limited)

## Agent A (Signals & Strategy)
1) Which 1–2 **short‑horizon signals** are most reliable for 1–5 day holds?
2) What are the most common **leakage pitfalls** in OHLCV features?
3) Should we optimize for **Sharpe or Sortino** for a small-capital demo?

## Agent B (Execution, Risk & Portfolio)
1) What is a reasonable **flat transaction cost** assumption for liquid U.S. equities?
2) What **max position size** and **daily loss cap** would you recommend for a small account?
3) What simple **kill‑switch** threshold is appropriate for a demo?

---

# What We Need Help With (Online Research by Agents)

Use these as targeted research tasks for your finance agents.

## Agent A — Signals & Strategy Research Tasks
- Find empirical studies on **signal half‑life** for equity momentum/mean‑reversion.
- Identify **feature families** with strong out‑of‑sample performance in recent literature.
- Gather references on **regime detection** methods used in systematic strategies.
- Summarize best practices for **avoiding leakage** in financial time series features.

## Agent B — Execution & Risk Research Tasks
- Compile realistic **transaction cost/slippage** benchmarks for U.S. equities and options.
- Find studies on **liquidity proxies** for backtests without full order books.
- Summarize effective **risk constraint frameworks** used in live trading systems.
- Identify best practices for **kill‑switch criteria** and live monitoring.

---

# Research Synthesis Template (for Agent Reports)

Use this structure when finance agents deliver findings so the plan can be updated consistently.

## Signals & Strategy Summary
- **Signal families with evidence:** (list signals, horizon, half‑life, and citations)
- **Feature transformations that improve stability:** (e.g., log returns, vol scaling)
- **Leakage pitfalls and mitigations:** (timestamping, corporate actions, lookahead)
- **Recommended baselines/benchmarks:** (buy‑and‑hold, equal‑weight, volatility‑targeted)
- **Primary risk‑adjusted metric(s):** (Sharpe/Sortino/Calmar rationale)
- **Regime detection methods:** (volatility, macro, clustering)
- **Cross‑asset or sector signals:** (if justified by evidence)
- **Non‑stationarity handling:** (rolling normalization, retraining cadence)
- **Signal validation protocol:** (IC, walk‑forward, bootstraps)
- **Signal half‑life thresholds:** (minimum horizon vs holding period)

## Execution, Risk & Portfolio Summary
- **Transaction cost model:** (bps by ADV, spread model, options assumptions)
- **Liquidity proxies:** (ADV, spreads, open interest)
- **Risk constraints:** (max drawdown, leverage, exposure caps)
- **Portfolio construction approach:** (risk parity, vol targeting, weight clipping)
- **Stress tests:** (shock scenarios and expected behavior)
- **Turnover constraints:** (target ranges and penalties)
- **Borrow/short constraints:** (costs and limits)
- **Live vs backtest drift checks:** (metrics + thresholds)
- **Retraining cadence:** (recommended schedule and triggers)
- **Kill‑switch criteria:** (explicit thresholds)

## Open Questions (Need Agent Input)
1) What is the most defensible **cost model** for daily equity trading with 1–5 day holding?
2) Which **regime features** consistently improve out‑of‑sample results?
3) What **turnover bounds** are realistic for our universe and execution stack?
4) Which **benchmarks** will satisfy stakeholders for performance validation?
5) What is the minimum **statistical threshold** to promote a model (e.g., Sharpe improvement, drawdown reduction)?
