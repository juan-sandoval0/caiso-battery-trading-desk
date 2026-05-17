# Handoff — CAISO Battery Storage Trading Desk

**Date:** 2026-05-17  
**Repo:** https://github.com/juan-sandoval0/caiso-battery-trading-desk  
**Phase complete:** Phase 1 (Data Pipeline)  
**Phase next:** Phase 2 (Price Forecaster + Dispatch Optimizer)

---

## What exists right now

### Fully implemented and tested
| File | What it does |
|------|-------------|
| `src/config/settings.py` | Pydantic settings loaded from `.env`; `get_settings()` is a cached singleton |
| `src/config/nodes.py` | CAISO hub node constants (`TH_NP15_GEN-APND`, SP15, ZP26) and `node_short_name()` |
| `src/config/battery.py` | `BatteryConfig` dataclass — 4 MWh, 1 MW, 87.5% round-trip efficiency |
| `src/data/db.py` | DuckDB wrapper with 8-table schema; `INSERT OR REPLACE` upserts; `query_lmp()`, `query_features()`, `get_latest_data_date()` |
| `src/data/caiso_fetcher.py` | Fetches LMP (DA + RT 5-min), load, fuel mix, storage SOC, curtailment via gridstatus; 30-day chunks, 3-retry backoff |
| `src/data/weather_fetcher.py` | Open-Meteo ERA5 historical + 7-day forecast; hourly solar irradiance, temperature, wind |
| `main.py` | `data-fetch` CLI with `--incremental`, `--skip-storage`, `--skip-weather` flags |

### Stub files (signatures + docstrings, no implementation)
All files in `src/agents/`, `src/orchestrator/`, `src/models/`, `src/optimization/`, `src/dashboard/`, and all `tests/` files exist with full type-hinted signatures and implementation comments. Nothing raises silently — everything is `raise NotImplementedError(...)`.

---

## How to run what works today

```bash
# Install deps
pip install -r requirements.txt

# Fetch historical data (run once; ~30-45 min for 3 years)
python main.py data-fetch --start 2023-02-01 --end 2026-04-30 --skip-storage --db data/market.duckdb

# Fetch incrementally after first run
python main.py data-fetch --start 2023-02-01 --end 2026-04-30 --skip-storage --db data/market.duckdb --incremental
```

---

## Phase 2 — what to build next

### 1. `src/models/features.py`
Build the feature matrix from DuckDB. Key features:
- LMP lags: `t-1h, t-2h, t-24h, t-48h, t-168h`
- Rolling stats: 4h, 24h, 168h mean and std
- Temporal: hour, day-of-week, month, is_weekend
- Solar: `shortwave_radiation`, `cloud_cover` from weather table
- Load: `load_mw` from load_actual table
- Entry point: `build_feature_matrix(lmp_df, load_df, weather_df)` → `(X, y)`

### 2. `src/models/price_model.py`
- `PriceModel(model_type='da')` wraps XGBoost; `model_type='rt'` wraps LightGBM
- `train(X_train, y_train, X_val, y_val)` → returns RMSE dict
- `save()` / `load()` use `joblib` to `models/artifacts/`
- `train_pipeline(node, start_train, end_train, ...)` ties DB → features → train → save
- Target RMSE: ≤ $15/MWh out-of-sample on NP15

### 3. `src/optimization/battery_dispatch.py`
- `BatteryDispatchOptimizer.solve(prices_mwh, interval_hours, initial_soc_mwh, constraints)`
- Pyomo `ConcreteModel` LP — no binary variables needed
- Solver: `appsi_highs` (install: `pip install highspy`); fallback `glpk`
- Also implement `solve_baseline_naive()` and `solve_perfect_hindsight()` for benchmarking
- `DispatchResult.to_dataframe(times)` for dashboard consumption

### 4. `main.py train` and `main.py backtest` commands
- `train`: calls `train_pipeline()`, prints val RMSE, saves artifact
- `backtest`: loads model + DB, iterates days, calls optimizer, writes P&L CSV

### Acceptance criteria for Phase 2
- [ ] DA forecaster RMSE ≤ $15/MWh on 2024 out-of-sample (NP15)
- [ ] LP solves to optimality in < 2s for a 24h horizon
- [ ] Backtest on 2024 shows positive P&L vs. naive valley-fill strategy
- [ ] SoC bounds (10%–90%) never violated in any solved schedule

---

## Key facts to remember

- **gridstatus columns (verified live):** `Time→time, Location→node, LMP→lmp, Energy→energy, Congestion→congestion, Loss→loss`
- **Storage SOC has a ~5-day data lag** — the fetcher handles 404s gracefully; use `--skip-storage` for recent date ranges
- **Curtailment PRIMARY KEY** is `(time, fuel_type, curtailment_type, reason)` — there are 6 distinct type/reason combos per interval
- **gridstatus logs noisily** via tqdm and its own root handler; this is cosmetic and not suppressible without patching the library
- **No CAISO auth needed** — all data is public via OASIS; gridstatus API key only increases rate limits
- **Open-Meteo rate limit** is 10,000 req/day — weather fetch for 3 years × 3 nodes = 3 requests total (one per node), well within limits

## Environment
- Python 3.12, gridstatus 0.36.0, DuckDB ≥ 1.0
- Known dependency conflict: `h11` version disagreement between gridstatus and httpcore — both work in practice, ignore the pip warning
- `.env` file needed for `ANTHROPIC_API_KEY` (Phase 3 only); all Phase 1–2 work runs without it
