# CAISO Battery Storage Trading Desk — Build Plan

## Guiding Principles

1. **Data first**: no agent is built before its data is verified to actually arrive
2. **Backtest before live**: every trading strategy is validated on 12+ months of history before paper-trading
3. **Fail loudly**: prefer `raise` over silent fallback; bad data should crash, not silently produce wrong P&L
4. **One solver, one truth**: Pyomo + HiGHS is the single dispatch decision-maker; agents only *advise* it via constraints

---

## Phase 1 — Data Pipeline

**Goal**: Reliable, queryable local store of CAISO LMPs + demand + generation + storage SOC

### Deliverables

| # | Deliverable | File(s) |
|---|-------------|---------|
| 1.1 | DuckDB schema + migration runner | `src/data/db.py` |
| 1.2 | CAISO LMP fetcher (DA hourly + RT 5-min) for NP15/SP15/ZP26 | `src/data/caiso_fetcher.py` |
| 1.3 | CAISO load, fuel mix, storage SOC fetchers | `src/data/caiso_fetcher.py` |
| 1.4 | Open-Meteo solar/wind forecast fetcher | `src/data/weather_fetcher.py` |
| 1.5 | CLI `data-fetch` command + incremental update logic | `main.py` |
| 1.6 | EDA notebook: price distributions, spreads, seasonality | `notebooks/01_eda.ipynb` |

### Acceptance Criteria

- [ ] Can fetch 30 days of 5-min RT LMPs for all three hubs without error
- [ ] DuckDB contains LMP, load, and weather tables; no duplicate rows
- [ ] Incremental fetch skips already-stored intervals
- [ ] EDA notebook renders price histogram showing typical NP15 spread ≥ $20/MWh on peak days

### Dependencies

- gridstatus 0.36 (installed)
- DuckDB ≥ 1.0
- `GRIDSTATUS_API_KEY` (optional but recommended for bulk fetches)

### Known Risks

- CAISO OASIS limits bulk requests; gridstatus throttles at ~5 s/request — budget ~10 min for a full year fetch
- gridstatus `get_lmp()` for `REAL_TIME_5_MIN` returns ~288 rows/day/node; verify column names before building schema
- Weather data from Open-Meteo is free with no key but has 10,000 req/day limit; cache aggressively

---

## Phase 2 — Core Agents: Price Forecaster + Dispatch Optimizer

**Goal**: End-to-end backtest pipeline producing daily dispatch schedule and P&L

### Deliverables

| # | Deliverable | File(s) |
|---|-------------|---------|
| 2.1 | Feature engineering: lags, rolling stats, calendar features, solar proxy | `src/models/features.py` |
| 2.2 | XGBoost DA-LMP forecaster (next 24 h, hourly) | `src/models/price_model.py` |
| 2.3 | LightGBM RT-LMP forecaster (next 12 intervals, 5-min) | `src/models/price_model.py` |
| 2.4 | Model training CLI + artifact persistence | `main.py`, `src/models/price_model.py` |
| 2.5 | Pyomo battery dispatch LP (charge/discharge schedule) | `src/optimization/battery_dispatch.py` |
| 2.6 | `PriceForecasterAgent` class wrapping models | `src/agents/forecaster.py` |
| 2.7 | `DispatchOptimizerAgent` class wrapping Pyomo | `src/agents/optimizer.py` |
| 2.8 | Backtester runner + P&L CSV output | `main.py` |

### Acceptance Criteria

- [ ] DA forecaster achieves RMSE ≤ $15/MWh on 2024 out-of-sample (NP15)
- [ ] Dispatch LP solves to optimality in < 2 s for a 24-h horizon
- [ ] Backtest on Jan–Dec 2024 shows net positive P&L vs. naive flat-price strategy
- [ ] Round-trip efficiency (87.5%), SoC bounds (10–90%), and ramp limits are enforced in all solutions

### Battery Parameters (virtual BESS)

```
Capacity:          4 MWh
Max charge power:  1 MW
Max discharge:     1 MW
Round-trip eff:    87.5%  (charge eff 93.5%, discharge eff 93.5%)
Degradation cost:  $0.05/kWh cycled
SoC bounds:        10% – 90%
Cycle limit:       1 full cycle/day
```

### Dependencies

- Phase 1 complete (historical LMP data in DuckDB)
- Pyomo + highspy installed
- XGBoost, LightGBM installed

---

## Phase 3 — Supporting Agents, Shared State, Orchestrator

**Goal**: Full multi-agent system running as a LangGraph graph; agents coordinate through SharedMarketState

### Deliverables

| # | Deliverable | File(s) |
|---|-------------|---------|
| 3.1 | `SharedMarketState` dataclass + DuckDB persistence | `src/orchestrator/shared_state.py` |
| 3.2 | `ConflictResolver` priority rules | `src/orchestrator/conflict_resolver.py` |
| 3.3 | LangGraph supervisor graph (tick loop) | `src/orchestrator/coordinator.py` |
| 3.4 | `RiskMonitorAgent`: spike/curtailment/emergency detection | `src/agents/risk_monitor.py` |
| 3.5 | `MarketIntelAgent`: Claude LLM parsing CAISO notices + weather alerts | `src/agents/market_intel.py` |
| 3.6 | Tool definitions for each agent (LangChain tools) | within each agent module |
| 3.7 | Integration test: full graph tick on historical snapshot | `tests/test_orchestrator/` |

### Agent Priority (conflict resolution)

```
1. RiskMonitor  — can HALT or OVERRIDE any dispatch; highest priority
2. MarketIntel  — can ADD constraints (e.g., "avoid discharge during EEA3")
3. Optimizer    — produces base schedule
4. Forecaster   — advisory; provides price signal to Optimizer
```

### LangGraph Graph Topology

```
START → [fetch_data] → [forecast_prices] → [run_optimization]
                                        ↓
                      [monitor_risk] ←→ [resolve_conflicts]
                                        ↓
                      [market_intel] → [finalize_dispatch] → END
```

### Acceptance Criteria

- [ ] Full graph tick completes in < 30 s on a single CPU core
- [ ] RiskMonitor correctly overrides optimizer on synthetic spike test (LMP > $500/MWh)
- [ ] MarketIntelAgent extracts correct curtailment constraint from a sample CAISO notice
- [ ] SharedMarketState serializes/deserializes without data loss

### Dependencies

- Phase 2 complete
- `ANTHROPIC_API_KEY` in `.env` for MarketIntelAgent
- LangGraph ≥ 0.2

---

## Phase 4 — Dashboard, Paper-Trading, Benchmarking, Report

**Goal**: Live operational system with real-time dashboard and rigorous performance benchmarking

### Deliverables

| # | Deliverable | File(s) |
|---|-------------|---------|
| 4.1 | Streamlit dashboard: live P&L, SoC gauge, price chart, agent log | `src/dashboard/app.py` |
| 4.2 | Baseline strategies: always-charge-at-night, perfect-hindsight upper bound | `src/optimization/battery_dispatch.py` |
| 4.3 | P&L benchmarking report (our system vs. baselines) | `notebooks/02_benchmarking.ipynb` |
| 4.4 | Paper-trading CLI mode (live data, simulated execution) | `main.py` |
| 4.5 | DigitalOcean deployment (optional: schedule daily fetch + backtest) | `deploy/` |
| 4.6 | Final project report | `report/final_report.md` |

### Baseline Strategies

| Strategy | Description |
|----------|-------------|
| Naive valley-fill | Charge 22:00–06:00, discharge 14:00–20:00 daily |
| DA-price optimal | Solve LP with perfect DA price knowledge (upper bound) |
| Random | Random charge/discharge within SoC bounds |
| Our system | Multi-agent forecaster + optimizer |

### Acceptance Criteria

- [ ] Dashboard auto-refreshes every 5 minutes in paper-trading mode
- [ ] Annual P&L of our system exceeds naive strategy by ≥ 15% on 2024 backtest
- [ ] All agent decisions are logged with timestamps and rationale
- [ ] Report includes Sharpe ratio, max drawdown, and win-rate by hour-of-day

### Dependencies

- Phases 1–3 complete
- Streamlit installed
- DigitalOcean API token (for deployment, optional)

---

## Timeline (Rough)

| Phase | Estimated effort | Target milestone |
|-------|-----------------|-----------------|
| Phase 1 | 1–2 days | Data pipeline verified |
| Phase 2 | 3–4 days | Backtest running, P&L positive |
| Phase 3 | 3–4 days | Full multi-agent graph |
| Phase 4 | 2–3 days | Dashboard live, report draft |

---

## Open Questions / Risks

1. **gridstatus rate limits**: bulk historical fetch for 3 nodes × 39 months × 5-min resolution = ~2M rows. May need to fetch in monthly chunks with sleep between requests.
2. **HiGHS availability**: `highspy` is the Python binding for HiGHS solver; confirm it installs cleanly (`pip install highspy`). Fallback: GLPK via `glpk` package.
3. **CAISO market notices**: no structured API exists. MarketIntelAgent will scrape `http://www.caiso.com/market/Pages/MarketNotices/Default.aspx` and pass HTML to Claude.
4. **LangGraph version compatibility**: API changes frequently; pin to a specific minor version once confirmed working.
5. **DigitalOcean $250 credits**: sufficient for a $6/month Droplet running the data fetch + backtest loop for the project duration. No GPU needed.
