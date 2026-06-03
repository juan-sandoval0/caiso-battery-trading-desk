# CAISO Battery Storage Trading Desk

A multi-agent AI platform that autonomously operates a simulated battery energy storage system (BESS) on California's CAISO electricity market. Agents collaborate to ingest real-time market data, forecast locational marginal prices (LMPs), solve constrained charge/discharge optimization, monitor risk, and parse qualitative market intelligence — all surfaced through a live P&L dashboard.

## Results (2024 H2 out-of-sample, NP15)

| Metric | Value |
|--------|-------|
| DA forecaster RMSE | **$15.61/MWh** (non-spike: $9.93) |
| Backtest net P&L | **+$1,054** vs. naive −$65,744 |
| Sharpe ratio | **1.49** |
| Optimizer solve time | ~0.012 s/day |
| SoC constraint violations | 0 of 176 days |
| Test suite | 82 passing |

## Architecture

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
│  XGBoost/LGBM     │  Pyomo + HiGHS      │  rule-based   │ Claude/OR │
│  DA + RT LMP      │  SoC-constrained    │  spike/curtail│  notices  │
└────────┬──────────────────────────────────────────────────────────┘
         │ reads
┌────────┴──────────────────────────────────────────────────────────┐
│  Data Layer: CAISO (gridstatus) + Weather + DuckDB local store    │
└───────────────────────────────────────────────────────────────────┘
```

**Data flow:**
1. `src/data/caiso_fetcher.py` pulls LMPs, load, generation, and storage SOC from CAISO OASIS via gridstatus
2. `src/data/weather_fetcher.py` pulls Open-Meteo solar/wind forecasts (free, no key required)
3. All raw data lands in DuckDB; agents query the DB and live feeds
4. `src/orchestrator/coordinator.py` runs a LangGraph graph that ticks on each 5-min interval
5. Final dispatch schedule is written to `SharedMarketState`; the Streamlit dashboard reads it live

## Project Structure

```
CS_153/
├── main.py                     # CLI: backtest | paper-trade | data-fetch | train
├── requirements.txt
├── .env.example
├── src/
│   ├── agents/
│   │   ├── forecaster.py       # PriceForecasterAgent (XGBoost/LightGBM)
│   │   ├── optimizer.py        # DispatchOptimizerAgent (Pyomo + HiGHS)
│   │   ├── risk_monitor.py     # RiskMonitorAgent (rule-based)
│   │   └── market_intel.py     # MarketIntelAgent (Claude via OpenRouter)
│   ├── orchestrator/
│   │   ├── shared_state.py     # SharedMarketState dataclass
│   │   ├── conflict_resolver.py# Priority rules: Risk > Optimizer > Forecaster
│   │   └── coordinator.py      # LangGraph supervisor graph
│   ├── data/
│   │   ├── caiso_fetcher.py    # gridstatus CAISO wrapper
│   │   ├── weather_fetcher.py  # Open-Meteo wrapper
│   │   └── db.py               # DuckDB read/write helpers
│   ├── models/
│   │   ├── features.py         # Feature engineering for LMP forecasting
│   │   └── price_model.py      # XGBoost/LightGBM train + inference
│   ├── optimization/
│   │   └── battery_dispatch.py # Pyomo MILP / LP battery dispatch model
│   ├── dashboard/
│   │   └── app.py              # Streamlit P&L dashboard
│   └── config/
│       ├── battery.py          # BESS physical + financial parameters
│       ├── nodes.py            # CAISO PNode / hub selection
│       └── settings.py         # Env-var loader, logging config
├── notebooks/
│   ├── 01_eda.ipynb            # Price distributions, spreads, seasonality
│   └── 02_benchmarking.ipynb   # Model and strategy benchmarks
├── report/
│   └── final_report.md
└── tests/
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | For live/paper-trade | `sk-or-v1-...` — routes Claude via OpenRouter |
| `GRIDSTATUS_API_KEY` | Optional | Increases CAISO OASIS rate limits |
| `DUCKDB_PATH` | Optional | Defaults to `data/market.duckdb` |
| `LOG_LEVEL` | Optional | `DEBUG \| INFO \| WARNING \| ERROR` |
| `DEFAULT_NODE` | Optional | Defaults to `TH_NP15_GEN-APND` |

### 3. Fetch historical data

```bash
python main.py data-fetch --start 2023-07-01 --end 2024-12-31 --skip-storage
```

This pulls DA/RT LMPs, load, fuel mix, and weather into DuckDB (~25 min for the full range).

### 4. Train the price forecasting model

```bash
python main.py train --node TH_NP15_GEN-APND --model-type da
python main.py train --node TH_NP15_GEN-APND --model-type rt
```

Trained artifacts are saved to `models/artifacts/`.

### 5. Run a backtest

```bash
python main.py backtest --start 2024-07-01 --end 2024-12-31 --node TH_NP15_GEN-APND
```

Results are written to `results/backtest_2024H2.csv`.

### 6. Run paper-trading (live mode)

```bash
python main.py paper-trade --node TH_NP15_GEN-APND
```

Requires `OPENROUTER_API_KEY` for the `MarketIntelAgent`.

### 7. Launch the dashboard

```bash
streamlit run src/dashboard/app.py
```

## BESS Parameters (virtual battery)

| Parameter | Value |
|-----------|-------|
| Capacity | 4 MWh |
| Max charge power | 1 MW |
| Max discharge power | 1 MW |
| Round-trip efficiency | 87.5% |
| SoC bounds | 10% – 90% |
| Degradation cost | $0.05 / kWh cycled |

## CAISO Nodes

| Hub | Node ID |
|-----|---------|
| NP15 (Northern CA) | `TH_NP15_GEN-APND` |
| SP15 (Southern CA) | `TH_SP15_GEN-APND` |
| ZP26 (Central CA) | `TH_ZP26_GEN-APND` |

## LLM Provider

The `MarketIntelAgent` calls Claude through **OpenRouter** (Anthropic-compatible API), so no Anthropic API key is needed. Set `OPENROUTER_API_KEY` in `.env`. The model used is `anthropic/claude-sonnet-4.6`.

## Testing

```bash
# Unit tests (fast, no network)
pytest tests/ -m "not integration" -v

# Integration tests (hit real CAISO OASIS)
RUN_INTEGRATION=1 pytest tests/ -v --timeout=120
```

82 unit tests pass; 2 integration tests are gated on `RUN_INTEGRATION=1`.
