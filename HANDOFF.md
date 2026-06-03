# Handoff — CAISO Battery Storage Trading Desk

**Date:** 2026-05-31 (revised; prior revisions 2026-05-30, 2026-05-17)
**Repo:** https://github.com/juan-sandoval0/caiso-battery-trading-desk
**Code state:** **All four phases complete**, committed on branch `finish-project` (9 commits ahead of `origin/main`, not yet pushed).
**Remaining:** push `finish-project` and open a PR (optional); optionally wire `ANTHROPIC_API_KEY` for a live MarketIntel paper-trade run.

> **History:** The 2026-05-17 handoff described everything past Phase 1 as
> `NotImplementedError` stubs; that was stale (code was written but uncommitted).
> A 2026-05-30 revision corrected that but predated the train/backtest/dashboard/
> notebook/report work. This revision reflects the finished project.

## TL;DR results (2024 H2 out-of-sample, NP15)
- DA forecaster RMSE **$15.61/MWh** full-H2 (MAE $6.42; $6.42 Oct–Dec). Summer spikes inflate it; non-spike RMSE $9.93.
- Backtest: net **+$1,054** vs naive **−$65,744**; Sharpe **1.49**; all 176 days solve optimal in ~0.012 s; SoC stays in [10%, 90%].
- Tests: **82 pass** (`pytest tests/ -m "not integration"`); 2 live-CAISO integration tests gated on `RUN_INTEGRATION=1`.
- Artifacts (gitignored): `models/artifacts/{da,rt}_NP15.joblib`, `results/backtest_2024H2.csv`.

---

## What exists right now

### Fully implemented (config + data layer, Phase 1)
| File | What it does |
|------|-------------|
| `src/config/settings.py` | Pydantic settings loaded from `.env`; `get_settings()` is a cached singleton |
| `src/config/nodes.py` | CAISO hub node constants (`TH_NP15_GEN-APND`, SP15, ZP26) and `node_short_name()` |
| `src/config/battery.py` | `BatteryConfig` dataclass — 4 MWh, 1 MW, 87.5% round-trip efficiency |
| `src/data/db.py` | DuckDB wrapper with 8-table schema; `INSERT OR REPLACE` upserts; `query_lmp()`, `query_features()`, `get_latest_data_date()` |
| `src/data/caiso_fetcher.py` | Fetches LMP (DA + RT 5-min), load, fuel mix, storage SOC, curtailment via gridstatus; 30-day chunks, 3-retry backoff |
| `src/data/weather_fetcher.py` | Open-Meteo ERA5 historical + 7-day forecast; hourly solar irradiance, temperature, wind |
| `main.py` | All four CLI commands wired: `data-fetch`, `train`, `backtest`, `paper-trade` |

### Implemented and validated against real data (Phases 2–4)
| Area | Files | Status |
|------|-------|--------|
| Forecasting | `src/models/features.py`, `src/models/price_model.py` | Trained on NP15; DA model fits in **log1p price space** (offset +150) |
| Optimization | `src/optimization/battery_dispatch.py` | Pyomo LP + naive/perfect baselines; ~0.012 s/solve |
| Agents | `src/agents/{forecaster,optimizer,risk_monitor,market_intel}.py` | Full impl |
| Orchestrator | `src/orchestrator/{shared_state,conflict_resolver,coordinator}.py` | LangGraph tick verified end-to-end |
| Dashboard | `src/dashboard/app.py` | Live + backtest pages; verified serving (health 200) |
| Notebooks/report | `notebooks/01_eda.ipynb`, `notebooks/02_benchmarking.ipynb`, `report/final_report.md` | Executed with embedded figures; report filled with real numbers |

### Data in DuckDB (`data/market.duckdb`, gitignored)
DA LMP (2023-02→2026-04, all 3 hubs), RT 5-min LMP / load / fuel mix / weather (2023-07→2024-12).
**curtailment and storage_soc are empty** — curtailment crashes inside gridstatus
(`ValueError: No objects to concatenate`); neither is used by `query_features` or any
downstream code, so this does not affect training/backtest.

---

## How to reproduce end-to-end

```bash
pip install -r requirements.txt
python main.py data-fetch --start 2023-07-01 --end 2024-12-31 --skip-storage   # ~25 min; curtailment step errors harmlessly
# weather is fetched inside data-fetch AFTER curtailment, so it may need a direct top-up:
#   python -c "from datetime import date; from src.data.db import MarketDB; from src.data.weather_fetcher import WeatherFetcher; from src.config.nodes import CAISO_HUB_NODES; \
#     db=MarketDB('data/market.duckdb'); wx=WeatherFetcher(); db.upsert_weather(wx.get_solar_history_all_nodes(CAISO_HUB_NODES, date(2023,7,1), date(2024,12,31)))"
python main.py train --node TH_NP15_GEN-APND --model-type da \
  --train-start 2023-07-01 --train-end 2024-06-30 --val-start 2024-07-01 --val-end 2024-09-30
python main.py train --node TH_NP15_GEN-APND --model-type rt \
  --train-start 2023-07-01 --train-end 2024-06-30 --val-start 2024-07-01 --val-end 2024-09-30
python main.py backtest --node TH_NP15_GEN-APND --start 2024-07-01 --end 2024-12-31 --output results/backtest_2024H2.csv
cd notebooks && jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb 02_benchmarking.ipynb && cd ..
pytest tests/ -v --timeout=60 -m "not integration"
streamlit run src/dashboard/app.py
```

---

## Gotchas discovered this session (don't re-trip them)

1. **`query_features` load join (fixed):** hourly DA LMP was joined to 5-min `load_actual` on the hour, fanning each hour into ~12 duplicate rows and silently corrupting all lag/rolling features. Now aggregates load to hourly. If you add another sub-hourly table to the join, aggregate it first.
2. **DA RMSE needs the log transform:** without `log1p` the full-H2 RMSE is ~$19.8; with it, $15.61. Summer heat-wave spikes (2.2% of hours, LMP→$583) are the only thing keeping it above $15 — they are not forecastable 24 h ahead from price history.
3. **Degradation cost is steep:** $0.05/kWh is charged on **both** charge and discharge legs (~$100/MWh round-trip), so the optimizer only trades on large spreads → thin profit (8/176 profitable days, 11.9% of perfect-hindsight). Tune `degradation_cost_per_kwh` in `src/config/battery.py` if you want more aggressive cycling.
4. **`data/` gitignore was unanchored** — it had hidden the entire `src/data/` package from git. Now anchored to `/data/`. Keep it anchored.
5. **`pytest-timeout` is required** by the documented `--timeout` test command; it's in `requirements.txt`.
6. **`paper-trade` needs `ANTHROPIC_API_KEY`** (MarketIntel/Claude) and current-dated CAISO data; it was validated only as a mocked single-tick graph test (`tests/test_orchestrator/test_coordinator_graph.py`), not a sustained live run.

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
