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
        cmd_train_stub(args)
    elif args.command == "backtest":
        cmd_backtest_stub(args)
    elif args.command == "paper-trade":
        cmd_paper_trade_stub(args)
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
# Phase 2 / 3 stubs (implemented later)
# ---------------------------------------------------------------------------

def cmd_train_stub(args: argparse.Namespace) -> None:
    print("Phase 2 not yet implemented. Run Phase 1 first: python main.py data-fetch ...")
    sys.exit(1)


def cmd_backtest_stub(args: argparse.Namespace) -> None:
    print("Phase 2 not yet implemented.")
    sys.exit(1)


def cmd_paper_trade_stub(args: argparse.Namespace) -> None:
    print("Phase 3 not yet implemented.")
    sys.exit(1)


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
    train.add_argument("--model-type", choices=["da", "rt"], default="da")
    train.add_argument("--db", default=None)

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

    return parser


if __name__ == "__main__":
    main()
