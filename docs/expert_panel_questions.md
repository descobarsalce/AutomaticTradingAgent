# Expert Panel Question Bank for Model Improvement

Below is a single-pass questionnaire for financial quants and ML researchers. Each question pairs a **context block** (how the current code behaves) with a **design doubt** we need resolved. The architecture today is PPO + Gym with direct OHLCV inputs, an in-memory portfolio, and Streamlit-driven orchestration; execution is single-process and assumes ideal liquidity. Use this detail to elicit concrete priors, thresholds, or redesign proposals instead of generic advice.

## System snapshot to anchor feedback
- **Policy/training loop:** Stable-Baselines3 PPO with an MLP policy, entropy regularization (`ent_coef=0.01`), optional deterministic seeding, and manifest logging. Actions are a per-asset Box in [-1,1] treated as buy/sell intensities.【F:core/base_agent.py†L200-L244】
- **Environment inputs:** `TradingEnv` consumes pre-shifted OHLCV from `TradingDataManager`; if no `feature_pipeline` is provided, it defaults to raw prices + positions + balance or the deprecated `feature_config`, leaving schema/versioning optional.【F:environment/trading_env.py†L53-L172】【F:data/data_handler.py†L20-L45】
- **Execution & sizing:** `PortfolioManager` executes sells, shuffles buys, and sizes trades from balance/position using a global max-%-per-asset cap. It ignores liquidity, borrow, or lot constraints and holds state in memory with no ledger or reconciliation layer.【F:core/portfolio_manager.py†L237-L295】
- **Orchestration/UI coupling:** Streamlit tabs stand up the agent, data handler, and training/inference lifecycle inside the UI process; there is no dedicated inference/exec service or message bus boundary.【F:main.py†L1-L200】

Use this snapshot to probe **first-order improvements** (feature parity, execution realism, risk correctness, reproducibility), not incremental latency tweaks.

## Data, features, and regime understanding
- **Data lineage and SLAs (design doubt: what makes data “admissible”?):** Ingestion uses a single provider with no freshness SLA or failover, and training assumes fully-shifted OHLCV is correct. What timestamping standard (exchange vs. UTC), freshness thresholds, redundancy policy, and per-batch metadata (provider, lag, adjustment flags) would you require before training or inference proceeds?【F:environment/trading_env.py†L55-L101】【F:data/data_handler.py†L20-L45】
- **Microstructure coverage (design doubt: minimal realism signals):** Observations omit bid/ask, depth, and halt flags. Which microstructure fields most improve fill realism for PPO, and how should we bucket or compress them to fit a fixed Box space without exploding dimensionality?
- **Regime/structure priors (design doubt: leakage-safe labeling):** Regimes are not encoded today. Which regime labels (vol/volume/trend buckets) have proven predictive, and how would you backfill/denoise them to avoid leakage when joined to shifted OHLCV?
- **Corporate action and roll adjustments (design doubt: adjustment surfacing):** Prices are assumed pre-adjusted. What adjustment stack (splits/dividends/rolls) is mandatory, and should adjustments be expressed as flags, masks, or price-only changes to keep PPO stable?
- **Feature validation gates (design doubt: admission tests):** With optional pipelines and no schema enforcement, what statistical gates (lookahead detection, PSI/KS drift, cross-asset independence) and thresholds should block a feature set from entering training?

## Execution realism and sizing
- **Liquidity-aware sizing (design doubt: mapping Box actions to fills):** `_calculate_trade_quantity` ignores ADV, spread, and lot sizes. What sizing model (participation caps, square-root impact, per-asset max notionals) would you impose so Box actions map to executable orders, and how should caps vary by asset class?【F:core/portfolio_manager.py†L240-L295】
- **Fill and slippage priors (design doubt: first-impact model):** Which microstructure/impact model (linear vs. concave, queue position) should we simulate first to avoid optimistic fills, and what minimal historical data is sufficient to calibrate it credibly for PPO rollouts?
- **Borrow and shortability (design doubt: feasibility checks):** How should we prevent infeasible shorts (borrow flags, hard-to-borrow lists) and incorporate borrow costs into rewards/action clipping so PPO respects real constraints?
- **Deterministic ordering & replay (design doubt: reproducible fills):** Trade processing shuffles buys with no recorded seed. What execution ordering (risk-reducing first, spread-ranked) and seed/manifest rules would you set so experiments and live trades can be replayed deterministically?【F:core/portfolio_manager.py†L248-L293】

## Risk controls and compliance
- **Portfolio guardrails (design doubt: enforceable limits):** With only balance/transaction-cost checks, which constraints (gross/net leverage, concentration, drawdown bands, VaR/ES) should run pre-trade vs. in-policy clipping? What thresholds are realistic for the assets we trade?【F:core/portfolio_manager.py†L15-L175】
- **Market state gating (design doubt: when to halt):** Data are annotated with availability, but the env trades every row. What calendar/halts logic would you require to block trading on closed or illiquid sessions, and what safe-mode behavior (flatten, hold, throttle) should trigger on data gaps?【F:data/data_handler.py†L20-L45】【F:environment/trading_env.py†L123-L185】
- **Auditability and ledgering (design doubt: crash recovery):** State is in-memory with no event log. What minimal append-only ledger schema and checkpoint cadence would satisfy institutional audit/recovery needs without blowing latency budgets?【F:core/portfolio_manager.py†L15-L124】【F:core/portfolio_manager.py†L166-L214】

## Evaluation, experimentation, and reproducibility
- **Train/test/live parity (design doubt: proof of sameness):** Feature/schema enforcement is optional. What hashing/versioning scheme for data slices, feature pipelines, and code would convince you that live and backtest observations are identical, and what manifest evidence is sufficient to prove parity?
- **Scenario coverage (design doubt: nightly stress set):** Which stress scenarios (flash crashes, limit-up/down, liquidity droughts) should be part of nightly evals, and how would you parameterize them to avoid overfitting PPO to one-off shocks?
- **Metrics that predict live success (design doubt: promotion signals):** Beyond Sharpe/drawdown, which metrics (turnover-adjusted decay, realized vs. simulated fill gap, risk-adjusted participation) best correlate with live success for PPO-style agents?
- **Reproducibility proofs (design doubt: determinism bar):** What seed-handling and determinism guarantees (ordering, RNG streams, artifact hashes) are sufficient before promoting a model, given current randomness in execution order?

## Learning architecture and objectives
- **Reward shaping and budgets (design doubt: objective that trades well):** How should we blend PnL with penalties for turnover/slippage/risk budget consumption to improve out-of-sample robustness? Are there reward formulations known to stabilize PPO under partial observability with OHLCV-only inputs?
- **Architecture choices (design doubt: representation depth):** With Box actions over multiple assets and shallow observations, would you favor PPO modifications (GAE tuning, entropy schedules, clipped value loss) or alternative actors/critics (distributional RL, recurrent encoders) to capture regime shifts without overfitting?
- **Exploration vs. exploitation (design doubt: safe exploration):** What exploration strategies (parameter noise, action smoothing, risk-aware noise) work in low-signal financial domains without breaking execution realism or risk limits?
- **Offline/online blend (design doubt: adaptation path):** How would you stage offline pretraining vs. online fine-tuning/paper trading to minimize live risk while adapting to new regimes?

## Deployment and operations
- **Service boundaries & SLIs (design doubt: decoupling plan):** With training and execution coupled to Streamlit, how would you decompose services (inference API, order router, queue, reconciliation) and which SLIs/SLOs (latency, freshness, risk headroom) should gate promotion to live trading?【F:main.py†L1-L200】
- **Drift and promotion gates (design doubt: go/no-go rules):** What drift tests (feature distribution, reward/metric deltas) and challenger/champion protocol would you require, and over what window, before swapping the production policy?
- **Incident response (design doubt: automation depth):** What on-call playbooks and automated mitigations (kill-switch triggers, position flattening, safe-mode data providers) should exist for data/feature/risk anomalies, and what telemetry is minimally necessary to support them?

Use these prompts to harvest concrete thresholds, data requirements, and modeling priors from the panel; map answers into hard gates (data quality), deterministic interfaces (features/actions), realistic simulators (liquidity/slippage), and service boundaries that decouple UI from execution.

_Last updated after incorporating all reviewer feedback; use as the single questionnaire version for expert outreach._
