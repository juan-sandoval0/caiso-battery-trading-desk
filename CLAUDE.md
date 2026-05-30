# CAISO Battery Storage Trading Desk — Claude Code Guide

## Project Overview

A multi-agent AI platform that autonomously operates a simulated battery energy storage system (BESS) on California's CAISO electricity market. Agents collaborate to ingest real-time market data, forecast locational marginal prices (LMPs), solve constrained charge/discharge optimization, monitor risk, and parse qualitative market intelligence — all surfaced through a live P&L dashboard.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Orchestrator                          │
│   coordinator.py — LangGraph supervisor resolving conflicts  │
│   shared_state.py — single source of truth for all agents   │
│   conflict_resolver.py — priority rules (Risk > Opt > FC)   │
└────────┬────────────────────────────────────────────────────┘
         │ reads / writes SharedMarketState (dataclass)
┌────────┴──────────────────────────────────────────────────────────┐
│  PriceForecaster  │  DispatchOptimizer  │  RiskMonitor  │  MktIntel │
│  XGBoost/LGBM     │  Pyomo + HiGHS      │  rule-based   │  Claude   │
│  DA + RT LMP      │  SoC-constrained    │  spike/curtail│  notices  │
└────────┬──────────────────────────────────────────────────────────┘
         │ reads
┌────────┴──────────────────────────────────────────────────────────┐
│  Data Layer: CAISO (gridstatus) + Weather + DuckDB local store    │
└───────────────────────────────────────────────────────────────────┘
```

**Data flow:**
1. `data/caiso_fetcher.py` pulls LMPs, load, generation, storage SOC from CAISO OASIS via gridstatus
2. `data/weather_fetcher.py` pulls Open-Meteo solar/wind forecasts (free, no key)
3. All raw data lands in DuckDB; agents query the DB + live feeds
4. `orchestrator/coordinator.py` runs a LangGraph graph that ticks on each new 5-min interval
5. Final dispatch schedule written back to `SharedMarketState`; Streamlit dashboard reads it

## Directory Structure

```
CS_153/
├── CLAUDE.md                   # this file
├── PLAN.md                     # phased delivery plan
├── main.py                     # CLI entry: backtest | paper-trade | data-fetch
├── requirements.txt
├── .env.example
├── data/                       # local DuckDB data store (gitignored)
├── src/
│   ├── agents/
│   │   ├── forecaster.py       # PriceForecasterAgent
│   │   ├── optimizer.py        # DispatchOptimizerAgent
│   │   ├── risk_monitor.py     # RiskMonitorAgent
│   │   └── market_intel.py     # MarketIntelAgent (Claude LLM)
│   ├── orchestrator/
│   │   ├── shared_state.py     # SharedMarketState dataclass + Redis/DuckDB sync
│   │   ├── conflict_resolver.py# priority rules + override logic
│   │   └── coordinator.py      # LangGraph supervisor graph
│   ├── data/
│   │   ├── caiso_fetcher.py    # gridstatus CAISO wrapper
│   │   ├── weather_fetcher.py  # Open-Meteo wrapper
│   │   └── db.py               # DuckDB read/write helpers
│   ├── models/
│   │   ├── features.py         # feature engineering for LMP forecasting
│   │   └── price_model.py      # XGBoost/LightGBM train + inference
│   ├── optimization/
│   │   └── battery_dispatch.py # Pyomo MILP / LP battery model
│   ├── dashboard/
│   │   └── app.py              # Streamlit dashboard
│   └── config/
│       ├── battery.py          # BESS physical + financial parameters
│       ├── nodes.py            # CAISO PNode / hub selection
│       └── settings.py         # env-var loader, logging config
└── tests/
    ├── test_agents/
    ├── test_orchestrator/
    ├── test_data/
    ├── test_models/
    └── test_optimization/
```

## Coding Conventions

- **Python 3.11+** with full type hints on every public function
- **Docstrings**: Google-style one-liner for simple functions, full Args/Returns for anything with >2 params
- **Async**: use `asyncio` for data fetch loops; agents expose both sync (`run()`) and async (`arun()`) interfaces
- **Logging**: `structlog` with JSON output; no bare `print()` in library code
- **Constants**: all magic numbers live in `src/config/`; never inline
- **No mutable globals** outside `SharedMarketState`
- Line length 100, formatted with `ruff format`

## Key Dependencies and Versions

| Package | Version | Purpose |
|---------|---------|---------|
| gridstatus | 0.36.0 | CAISO data (LMP, load, storage SOC, curtailment) |
| anthropic | ≥0.40 | Claude API (market intel agent) |
| langchain | ≥0.3 | Agent tool-calling primitives |
| langgraph | ≥0.2 | Supervisor orchestration graph |
| pyomo | ≥6.7 | Battery dispatch MILP model |
| highspy | latest | HiGHS LP/MILP solver (open-source) |
| xgboost | ≥2.0 | LMP price forecasting |
| lightgbm | ≥4.0 | Alternative price forecasting model |
| duckdb | ≥1.0 | Local columnar data store |
| streamlit | ≥1.31 | Dashboard |
| pandas | ≥2.2 | Data wrangling |
| structlog | ≥24 | Structured logging |
| pytest | ≥8 | Testing |
| pytest-asyncio | ≥0.23 | Async test support |

## CAISO Data Details

- **Hub nodes**: `TH_NP15_GEN-APND`, `TH_SP15_GEN-APND`, `TH_ZP26_GEN-APND`
- **Markets**: `REAL_TIME_5_MIN`, `REAL_TIME_15_MIN`, `DAY_AHEAD_HOURLY`
- **History**: ~39 months available via CAISO OASIS (no auth required)
- **gridstatus API key**: optional for higher rate limits; set `GRIDSTATUS_API_KEY` in `.env`
- **Storage methods**: `get_storage_soc_rtd()`, `get_storage_awards_rtd()`, `get_storage_energy_bids_ifm()`

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill environment variables
cp .env.example .env

# 3. Fetch historical data (one-time or incremental)
python main.py data-fetch --start 2023-01-01 --end 2024-12-31

# 4. Train price forecasting model
python main.py train --node TH_NP15_GEN-APND

# 5. Run backtest
python main.py backtest --start 2024-01-01 --end 2024-12-31 --node TH_NP15_GEN-APND

# 6. Run paper-trading (live mode against real-time CAISO data)
python main.py paper-trade --node TH_NP15_GEN-APND

# 7. Launch dashboard
streamlit run src/dashboard/app.py
```

## Testing Strategy

- **Unit tests**: each agent and module tested in isolation with mocked data fetchers
- **Integration tests**: data pipeline tested against real CAISO API with a short date range
- **Backtest regression**: a reference backtest result (P&L CSV) is committed; CI asserts it matches within tolerance
- Run tests: `pytest tests/ -v --timeout=60`
- Integration tests are marked `@pytest.mark.integration` and skipped in CI unless `RUN_INTEGRATION=1`

## Current Status

- [x] Phase 1: Data pipeline (NP15/SP15/ZP26, 2023-07–2024-12 in DuckDB)
- [x] Phase 2: Core agents (Forecaster RMSE $15.61 full-H2 OOS; Optimizer LP < 0.02 s)
- [x] Phase 3: Supporting agents + Orchestrator (LangGraph tick verified end-to-end)
- [x] Phase 4: Dashboard + benchmarking notebooks + final report
- Tests: 82 passing (`pytest tests/ -m "not integration"`); 2 live-CAISO integration tests gated on `RUN_INTEGRATION=1`
- Backtest (2024 H2 OOS): net +$1,054 vs naive −$65,744; Sharpe 1.49; SoC bounds respected on all 176 days
