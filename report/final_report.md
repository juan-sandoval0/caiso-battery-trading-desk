# CAISO Battery Storage Trading Desk — Final Report

**Course:** CS 153
**Author:** Juan Sandoval
**Date:** 2026-05-30

> _Numeric results below are populated from the 2024 H2 out-of-sample backtest
> (`results/backtest_2024H2.csv`) and the trained NP15 models. Placeholders
> marked `<<...>>` are filled from the executed pipeline._

---

## 1. Overview

This project is a multi-agent AI platform that autonomously operates a simulated
battery energy storage system (BESS) on California's CAISO electricity market.
Four specialized agents — a **price forecaster**, a **dispatch optimizer**, a
**risk monitor**, and an **LLM-based market-intelligence agent** — collaborate
through a shared state object, coordinated by a LangGraph supervisor graph, to
ingest market data, forecast locational marginal prices (LMPs), solve a
constrained charge/discharge optimization, and guard against market anomalies.

The virtual battery is a 4 MWh / 1 MW unit with 87.5% round-trip efficiency,
SoC bounds of 10–90%, a $0.05/kWh degradation cost, and a one-cycle-per-day
limit (`src/config/battery.py`).

## 2. Architecture

```
Orchestrator (LangGraph)
  fetch → market_intel → forecast → risk → optimize → resolve → finalize
        SharedMarketState (immutable snapshot per tick)
Agents: PriceForecaster (XGBoost/LightGBM) · DispatchOptimizer (Pyomo + HiGHS)
        RiskMonitor (rule-based) · MarketIntel (Claude)
Data:   CAISO via gridstatus + Open-Meteo weather → DuckDB
```

Conflict-resolution priority: **RiskMonitor > MarketIntel > Optimizer > Forecaster**.
The RiskMonitor can HALT all dispatch (e.g., on an EEA3 grid emergency) or inject
constraints (halt discharge on a price spike, force charge on negative prices);
these are merged and applied by `src/orchestrator/conflict_resolver.py`.

## 3. Methodology

### 3.1 Data
- ~15 months of CAISO data (2023-07 → 2024-12) for NP15/SP15/ZP26: day-ahead
  hourly + real-time 5-min LMP, system load, fuel mix, curtailment, plus
  Open-Meteo solar/weather. Stored in DuckDB (`src/data/db.py`).

### 3.2 Forecasting
- Features (`src/models/features.py`): LMP lags (1–168 h), rolling mean/std
  (4 h / 24 h / 168 h), calendar features, load, and solar irradiance.
- DA model: XGBoost; RT model: LightGBM (`src/models/price_model.py`).
- **DA out-of-sample RMSE (NP15): `<<DA_RMSE>>` $/MWh** (target ≤ $15/MWh).

### 3.3 Dispatch optimization
- Continuous LP over a 24 h horizon (`src/optimization/battery_dispatch.py`),
  maximizing arbitrage revenue net of degradation, subject to SoC dynamics,
  power limits, SoC bounds, and the daily cycle limit. Solved with HiGHS.
- Solve time: **`<<SOLVE_TIME>>` s** per 24 h horizon (target < 2 s).

## 4. Backtest Results (2024 H2, out-of-sample, NP15)

| Strategy | Total revenue |
|----------|--------------:|
| Our multi-agent system | `<<OUR_REV>>` |
| Naive valley-fill | `<<NAIVE_REV>>` |
| Perfect hindsight (upper bound) | `<<PERFECT_REV>>` |

- **vs. naive: `<<VS_NAIVE_PCT>>`%** (target ≥ +15%)
- **% of perfect-hindsight bound: `<<PCT_PERFECT>>`%**
- Annualized Sharpe: `<<SHARPE>>` · Max drawdown: `<<MAX_DD>>` · Win rate: `<<WIN_RATE>>`%
- SoC bounds (10–90%) and the cycle limit were respected on every solved day.

See `notebooks/02_benchmarking.ipynb` for the full figures.

## 5. Risk & Market-Intelligence Agents

- **RiskMonitor** detects price spikes (> $500/MWh → halt discharge), negative
  prices (< −$50/MWh → force charge), intra-interval ramps, EEA1/2/3 alerts
  (EEA3 → full HALT), and SoC-trajectory violations.
- **MarketIntel** scrapes CAISO market notices and NWS weather alerts and uses
  Claude to extract a structured summary (EEA level, curtailment, outages,
  dispatch recommendation) consumed by the RiskMonitor and optimizer.

## 6. Limitations & Future Work

- The degradation cost is applied to both charge and discharge legs (~$100/MWh
  round-trip throughput), which suppresses arbitrage on small spreads; a
  cycle-aware degradation model would be more accurate.
- The RT (5-min) forecaster uses hourly features as a proxy; a native 5-min
  feature pipeline would improve intra-hour dispatch.
- Paper-trading was validated as a single-tick smoke test, not a sustained live run.
- No transmission/ancillary-service co-optimization; energy arbitrage only.

## 7. Reproducibility

```bash
pip install -r requirements.txt
python main.py data-fetch --start 2023-07-01 --end 2024-12-31 --skip-storage
python main.py train --node TH_NP15_GEN-APND --model-type da \
  --train-start 2023-07-01 --train-end 2024-06-30 \
  --val-start 2024-07-01 --val-end 2024-09-30
python main.py backtest --node TH_NP15_GEN-APND \
  --start 2024-07-01 --end 2024-12-31 --output results/backtest_2024H2.csv
pytest tests/ -v --timeout=60 -m "not integration"
streamlit run src/dashboard/app.py
```
