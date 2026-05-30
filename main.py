"""
CAISO Battery Storage Trading Desk — CLI entry point.

Commands:
    data-fetch    Pull historical CAISO + weather data into DuckDB
    train         Train LMP forecasting models  [Phase 2]
    backtest      Run strategy backtest          [Phase 2]
    paper-trade   Live paper-trading loop        [Phase 3]

Usage:
    python main.py data-fetch --start 2024-01-01 --end 2024-12-31
    python main.py data-fetch --start 2024-01-01 --end 2024-12-31 --incremental
    python main.py data-fetch --start 2024-01-01 --end 2024-12-31 --skip-storage
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from src.config.nodes import CAISO_HUB_NODES, DEFAULT_HUB_NODE
from src.config.settings import configure_logging, get_settings


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate command handler."""
    parser = _build_parser()
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    if args.command == "data-fetch":
        cmd_data_fetch(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "paper-trade":
        cmd_paper_trade(args)
    else:
        parser.print_help()
        sys.exit(1)


# ---------------------------------------------------------------------------
# data-fetch
# ---------------------------------------------------------------------------

def cmd_data_fetch(args: argparse.Namespace) -> None:
    """Pull historical CAISO + weather data into DuckDB."""
    import logging
    from datetime import datetime

    from src.data.caiso_fetcher import CAISOFetcher
    from src.data.db import MarketDB
    from src.data.weather_fetcher import WeatherFetcher

    log = logging.getLogger(__name__)

    settings = get_settings()
    db_path = args.db or str(settings.duckdb_path)
    nodes = args.nodes or CAISO_HUB_NODES

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print(f"\n{'='*60}")
    print(f"  CAISO Data Fetch")
    print(f"  Range : {start} → {end}")
    print(f"  Nodes : {nodes}")
    print(f"  DB    : {db_path}")
    print(f"{'='*60}\n")

    with MarketDB(db_path) as db:
        caiso = CAISOFetcher(api_key=settings.gridstatus_api_key)

        # ------------------------------------------------------------------
        # Determine effective start for incremental mode
        # ------------------------------------------------------------------
        if args.incremental:
            latest = db.get_latest_data_date("lmp")
            if latest:
                incremental_start = date.fromisoformat(latest) + timedelta(days=1)
                if incremental_start > end:
                    print("Database is already up to date. Nothing to fetch.")
                    return
                print(f"[incremental] Resuming from {incremental_start}\n")
                start = incremental_start

        # ------------------------------------------------------------------
        # DA LMP
        # ------------------------------------------------------------------
        print("▶ Fetching DA LMP (hourly)...")
        da_df = caiso.get_lmp_history(start, end, market="DAY_AHEAD_HOURLY", nodes=nodes)
        n = db.upsert_lmp(da_df)
        print(f"  ✓ DA LMP: {len(da_df):,} rows fetched, {n:+d} net new rows in DB\n")

        # ------------------------------------------------------------------
        # RT 5-min LMP
        # ------------------------------------------------------------------
        print("▶ Fetching RT 5-min LMP...")
        rt_df = caiso.get_lmp_history(start, end, market="REAL_TIME_5_MIN", nodes=nodes)
        n = db.upsert_lmp(rt_df)
        print(f"  ✓ RT LMP: {len(rt_df):,} rows fetched, {n:+d} net new rows in DB\n")

        # ------------------------------------------------------------------
        # System load (actual)
        # ------------------------------------------------------------------
        print("▶ Fetching system load (actual)...")
        load_df = caiso.get_load_history(start, end)
        n = db.upsert_load_actual(load_df)
        print(f"  ✓ Load actual: {len(load_df):,} rows fetched, {n:+d} net new rows in DB\n")

        # ------------------------------------------------------------------
        # Fuel mix
        # ------------------------------------------------------------------
        print("▶ Fetching fuel mix...")
        fm_df = caiso.get_fuel_mix_history(start, end)
        n = db.upsert_fuel_mix(fm_df)
        print(f"  ✓ Fuel mix: {len(fm_df):,} rows fetched, {n:+d} net new rows in DB\n")

        # ------------------------------------------------------------------
        # Curtailment
        # ------------------------------------------------------------------
        print("▶ Fetching curtailment...")
        curt_df = caiso.get_curtailment_history(start, end)
        n = db.upsert_curtailment(curt_df)
        print(f"  ✓ Curtailment: {len(curt_df):,} rows fetched, {n:+d} net new rows in DB\n")

        # ------------------------------------------------------------------
        # Storage SOC (skippable; has 5-day lag, many 404s)
        # ------------------------------------------------------------------
        if not args.skip_storage:
            print("▶ Fetching storage SOC (day-by-day, may be slow)...")
            soc_df = caiso.get_storage_soc_history(start, end)
            n = db.upsert_storage_soc(soc_df)
            print(f"  ✓ Storage SOC: {len(soc_df):,} rows fetched, {n:+d} net new rows in DB\n")
        else:
            print("  [skipped] Storage SOC (--skip-storage)\n")

        # ------------------------------------------------------------------
        # Weather (Open-Meteo historical)
        # ------------------------------------------------------------------
        if not args.skip_weather:
            print("▶ Fetching weather (Open-Meteo ERA5)...")
            with WeatherFetcher() as wx:
                wx_df = wx.get_solar_history_all_nodes(nodes, start, end)
            n = db.upsert_weather(wx_df)
            print(f"  ✓ Weather: {len(wx_df):,} rows fetched, {n:+d} net new rows in DB\n")
        else:
            print("  [skipped] Weather (--skip-weather)\n")

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        counts = db.row_counts()
        print(f"{'='*60}")
        print("  Database row counts after fetch:")
        for table, count in counts.items():
            print(f"    {table:<20} {count:>10,}")
        print(f"{'='*60}\n")
        print("Done.")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    """Train an LMP forecasting model on historical DuckDB data."""
    from datetime import date

    from src.config.nodes import node_short_name
    from src.data.db import MarketDB
    from src.models.price_model import train_pipeline

    settings = get_settings()
    db_path = args.db or str(settings.duckdb_path)
    node = args.node
    model_type = args.model_type

    # Determine date ranges: use explicit args or auto-detect from DB
    with MarketDB(db_path) as db:
        latest = db.get_latest_data_date("lmp")
        earliest_result = db.execute(
            "SELECT MIN(time) FROM lmp WHERE node = ? AND market = 'DAY_AHEAD_HOURLY'",
            [node],
        )
        earliest_row = earliest_result.iloc[0, 0]
        earliest = str(earliest_row)[:10] if earliest_row is not None else None

    if earliest is None or latest is None:
        print("No LMP data in DB. Run: python main.py data-fetch --start ... --end ... first.")
        sys.exit(1)

    if args.train_start:
        start_train = args.train_start
    else:
        start_train = earliest

    if args.train_end:
        end_train = args.train_end
    else:
        # Default: train on all but last 3 months
        from datetime import timedelta
        end_train = str(date.fromisoformat(latest) - timedelta(days=90))

    if args.val_start:
        start_val = args.val_start
    else:
        from datetime import timedelta
        start_val = str(date.fromisoformat(end_train) + timedelta(days=1))

    if args.val_end:
        end_val = args.val_end
    else:
        from datetime import timedelta
        end_val = str(date.fromisoformat(latest) - timedelta(days=30))

    short = node_short_name(node)
    print(f"\n{'='*60}")
    print(f"  Train {model_type.upper()} Model")
    print(f"  Node    : {node}")
    print(f"  Train   : {start_train} → {end_train}")
    print(f"  Val     : {start_val} → {end_val}")
    print(f"  DB      : {db_path}")
    print(f"{'='*60}\n")

    model = train_pipeline(
        node=node,
        start_train=start_train,
        end_train=end_train,
        start_val=start_val,
        end_val=end_val,
        model_type=model_type,
        db_path=db_path,
    )

    # Print feature importances
    try:
        top = model.feature_importance(top_n=10)
        print("\nTop-10 Feature Importances:")
        for _, row in top.iterrows():
            print(f"  {row['feature']:<35} {row['importance']:.4f}")
    except Exception:
        pass

    print(f"\nModel saved to: models/artifacts/{model_type}_{short}.joblib")
    print("Done.\n")


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

def cmd_backtest(args: argparse.Namespace) -> None:
    """Run a day-by-day backtest of the trained model + dispatch optimizer."""
    from datetime import date as _date, timedelta
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from src.agents.optimizer import DispatchOptimizerAgent
    from src.config.battery import DEFAULT_BATTERY
    from src.config.nodes import node_short_name
    from src.data.db import MarketDB
    from src.models.features import build_feature_matrix
    from src.models.price_model import PriceModel

    settings = get_settings()
    db_path = args.db or str(settings.duckdb_path)
    node = args.node
    short = node_short_name(node)

    # Locate trained model artifact
    model_path = Path("models/artifacts") / f"da_{short}.joblib"
    if not model_path.exists():
        print(f"Model not found at {model_path}.")
        print(f"Run: python main.py train --node {node}")
        sys.exit(1)
    model = PriceModel.load(model_path, model_type="da")

    start = _date.fromisoformat(args.start)
    end = _date.fromisoformat(args.end)

    # Need 10 days of lookback for 168h lag features
    lookback_start = str(start - timedelta(days=10))

    print(f"\n{'='*60}")
    print(f"  Backtest")
    print(f"  Node  : {node}")
    print(f"  Range : {start} → {end}")
    print(f"  Model : {model_path}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")

    with MarketDB(db_path) as db:
        features_df = db.query_features(node, lookback_start, str(end))

    if features_df.empty:
        print("No feature data in DB. Run data-fetch first.")
        sys.exit(1)

    print(f"Building feature matrix from {len(features_df)} rows...")
    X_all, y_all = build_feature_matrix(
        features_df, pd.DataFrame(), pd.DataFrame(), forecast_horizon_h=24
    )
    print(f"  {len(X_all)} feature rows, {X_all.shape[1]} features.\n")

    agent = DispatchOptimizerAgent(battery=DEFAULT_BATTERY, node=node)
    daily_results: list[dict] = []

    current = start
    n_skipped = 0
    while current <= end:
        prev_date = current - timedelta(days=1)

        # Filter feature rows for the previous day (they predict today's prices)
        idx = X_all.index
        if hasattr(idx, "tz") and idx.tz is not None:
            day_dates = np.array([ts.date() for ts in idx])
        else:
            day_dates = np.array([pd.Timestamp(ts).date() for ts in idx])

        mask_x = day_dates == prev_date
        mask_y = day_dates == prev_date

        X_day = X_all[mask_x]
        y_day = y_all[mask_y]

        if len(X_day) < 24 or len(y_day) < 24:
            n_skipped += 1
            current += timedelta(days=1)
            continue

        X_day = X_day.iloc[:24]
        y_day = y_day.iloc[:24]

        forecast_prices = model.predict(X_day)
        actual_prices = y_day.values

        row = agent.run_backtest_day(
            date=str(current),
            actual_prices=actual_prices,
            forecast_prices=forecast_prices,
        )
        daily_results.append(row)
        current += timedelta(days=1)

    if n_skipped > 0:
        print(f"  [info] Skipped {n_skipped} days with insufficient data.\n")

    if not daily_results:
        print("No backtest results. Check date range and data availability.")
        sys.exit(1)

    results_df = pd.DataFrame(daily_results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    total_rev = results_df["revenue_usd"].sum()
    total_naive = results_df["naive_revenue_usd"].sum()
    total_perfect = results_df["perfect_revenue_usd"].sum()
    avg_rmse = results_df["forecast_rmse"].mean()
    n_days = len(results_df)
    pct_vs_naive = ((total_rev - total_naive) / abs(total_naive) * 100) if total_naive != 0 else float("nan")
    pct_vs_perfect = (total_rev / total_perfect * 100) if total_perfect != 0 else float("nan")

    print(f"{'='*60}")
    print(f"  Backtest Results: {args.start} → {args.end}")
    print(f"  Days            : {n_days}")
    print(f"{'='*60}")
    print(f"  Our Revenue     : ${total_rev:>10,.2f}")
    print(f"  Naive Revenue   : ${total_naive:>10,.2f}")
    print(f"  Perfect Bound   : ${total_perfect:>10,.2f}")
    print(f"  vs Naive        : ${total_rev - total_naive:>+10,.2f}  ({pct_vs_naive:+.1f}%)")
    print(f"  vs Perfect      : ${total_rev - total_perfect:>+10,.2f}  ({pct_vs_perfect:.1f}% of bound)")
    print(f"  Daily Avg       : ${total_rev / n_days:>10,.2f}")
    print(f"  Forecast RMSE   : ${avg_rmse:>10.2f}/MWh")
    print(f"  Results saved   : {output_path}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# paper-trade (Phase 3)
# ---------------------------------------------------------------------------

def cmd_paper_trade(args: argparse.Namespace) -> None:
    """Run the live multi-agent paper-trading loop against real-time CAISO data."""
    import asyncio

    from src.orchestrator.coordinator import TradingDeskCoordinator

    settings = get_settings()
    if not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env — required for MarketIntelAgent.")
        sys.exit(1)

    node = getattr(args, "node", settings.default_node)
    interval = getattr(args, "interval", 5)

    print(f"\n{'='*60}")
    print(f"  Paper Trading — Live Mode")
    print(f"  Node     : {node}")
    print(f"  Interval : {interval} min")
    print(f"{'='*60}")
    print("  Starting coordinator… press Ctrl+C to stop.\n")

    coordinator = TradingDeskCoordinator.from_config(settings)
    asyncio.run(coordinator.run_live(node=node, interval_minutes=interval))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CAISO Battery Storage Trading Desk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # data-fetch
    fetch = sub.add_parser("data-fetch", help="Pull historical CAISO + weather data")
    fetch.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    fetch.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    fetch.add_argument(
        "--nodes", nargs="+", default=None,
        help="CAISO PNode IDs (default: all 3 hubs)",
    )
    fetch.add_argument("--db", default=None, help="DuckDB path (overrides DUCKDB_PATH env)")
    fetch.add_argument(
        "--incremental", action="store_true",
        help="Resume from the last stored date instead of re-fetching",
    )
    fetch.add_argument(
        "--skip-storage", action="store_true",
        help="Skip storage SOC (useful when fetching recent dates within the 5-day lag window)",
    )
    fetch.add_argument(
        "--skip-weather", action="store_true",
        help="Skip Open-Meteo weather fetch",
    )

    # train (Phase 2)
    train = sub.add_parser("train", help="Train LMP forecasting models [Phase 2]")
    train.add_argument("--node", default=DEFAULT_HUB_NODE)
    train.add_argument("--model-type", choices=["da", "rt"], default="da", dest="model_type")
    train.add_argument("--db", default=None)
    train.add_argument("--train-start", default=None, dest="train_start",
                       help="Training start date YYYY-MM-DD (auto-detects from DB if omitted)")
    train.add_argument("--train-end", default=None, dest="train_end",
                       help="Training end date YYYY-MM-DD (default: 90 days before latest data)")
    train.add_argument("--val-start", default=None, dest="val_start",
                       help="Validation start date (default: day after train-end)")
    train.add_argument("--val-end", default=None, dest="val_end",
                       help="Validation end date (default: 30 days before latest data)")

    # backtest (Phase 2)
    bt = sub.add_parser("backtest", help="Run strategy backtest [Phase 2]")
    bt.add_argument("--start", required=True)
    bt.add_argument("--end", required=True)
    bt.add_argument("--node", default=DEFAULT_HUB_NODE)
    bt.add_argument("--db", default=None)
    bt.add_argument("--output", default="results/backtest.csv")

    # paper-trade (Phase 3)
    pt = sub.add_parser("paper-trade", help="Live paper-trading [Phase 3]")
    pt.add_argument("--node", default=DEFAULT_HUB_NODE)
    pt.add_argument("--db", default=None)
    pt.add_argument(
        "--interval", type=int, default=5,
        help="Tick interval in minutes (default: 5, matching CAISO RT dispatch)",
    )

    return parser


if __name__ == "__main__":
    main()
