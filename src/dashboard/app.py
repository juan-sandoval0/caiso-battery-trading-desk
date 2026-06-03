"""
Streamlit dashboard for the CAISO Battery Storage Trading Desk.

Displays real-time and backtested system state including:
    - Live LMP chart for NP15/SP15/ZP26 hubs (auto-refreshing every 5 min)
    - Battery SoC gauge and charge/discharge bar chart
    - Forecasted vs. actual LMP overlay
    - Daily / cumulative P&L vs. baseline strategies
    - Risk event log with timestamps and rationale
    - Agent decision audit trail
    - Backtest summary statistics (RMSE, Sharpe, max drawdown)

Run with:
    streamlit run src/dashboard/app.py

The dashboard connects to the local DuckDB database for historical data
and optionally to a backtest results CSV written by `main.py backtest`.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config.battery import DEFAULT_BATTERY
from src.config.nodes import CAISO_HUB_NODES, DEFAULT_HUB_NODE

# DuckDB path (override via the sidebar)
_DEFAULT_DB_PATH: Path = Path("data/market.duckdb")

# Default backtest results CSV produced by `python main.py backtest`
_DEFAULT_BACKTEST_CSV: Path = Path("results/backtest_2024H2.csv")

# Auto-refresh interval in seconds (matches CAISO 5-min RT dispatch)
_REFRESH_INTERVAL_S: int = 300


def main() -> None:
    """Entry point for the Streamlit dashboard."""
    st.set_page_config(page_title="CAISO Battery Trading Desk", layout="wide")
    st.title("🔋 CAISO Battery Storage Trading Desk")

    config = render_sidebar()

    if config["mode"] == "backtest":
        _render_backtest_page(config)
    else:
        _render_live_page(config)


def render_sidebar() -> dict[str, Any]:
    """Render the sidebar with mode selection and configuration.

    Returns:
        Dict with keys: 'mode' ('live' | 'backtest'), 'node', 'db_path',
        'backtest_csv'.
    """
    st.sidebar.title("Settings")
    mode_label = st.sidebar.radio("Mode", ["Live / Paper-trade", "Backtest"])
    mode = "backtest" if mode_label == "Backtest" else "live"

    node = st.sidebar.selectbox(
        "Hub node",
        CAISO_HUB_NODES,
        index=CAISO_HUB_NODES.index(DEFAULT_HUB_NODE) if DEFAULT_HUB_NODE in CAISO_HUB_NODES else 0,
    )
    db_path = st.sidebar.text_input("DuckDB path", value=str(_DEFAULT_DB_PATH))
    backtest_csv = st.sidebar.text_input("Backtest CSV", value=str(_DEFAULT_BACKTEST_CSV))

    if mode == "live":
        st.sidebar.caption(f"Auto-refreshing every {_REFRESH_INTERVAL_S // 60} min.")
        # Lightweight meta-refresh; avoids an extra autorefresh dependency.
        st.markdown(
            f'<meta http-equiv="refresh" content="{_REFRESH_INTERVAL_S}">',
            unsafe_allow_html=True,
        )

    return {"mode": mode, "node": node, "db_path": Path(db_path), "backtest_csv": Path(backtest_csv)}


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #

def _render_live_page(config: dict[str, Any]) -> None:
    """Render the live / paper-trade page from DuckDB state."""
    node = config["node"]
    db = _load_db(config["db_path"])
    if db is None:
        st.warning(f"No DuckDB database found at `{config['db_path']}`. Run `main.py data-fetch` first.")
        return

    decisions = _load_agent_decisions(db)
    current_soc = _latest_soc_mwh(decisions)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Recent LMP — {node}")
        lmp_df = _load_recent_lmp(db, node)
        render_lmp_chart(lmp_df)
    with col2:
        st.subheader("Battery SoC")
        render_soc_gauge(current_soc, DEFAULT_BATTERY.capacity_mwh)

    st.subheader("Risk events & agent decisions")
    render_risk_event_log(decisions)


def _render_backtest_page(config: dict[str, Any]) -> None:
    """Render the backtest page from a results CSV."""
    csv_path = config["backtest_csv"]
    if not Path(csv_path).exists():
        st.warning(
            f"No backtest results at `{csv_path}`. "
            f"Run `python main.py backtest --output {csv_path}` first."
        )
        return

    raw = pd.read_csv(csv_path)
    pnl_df = _normalize_backtest_df(raw)

    st.subheader("Backtest summary")
    render_backtest_summary(pnl_df)

    st.subheader("Cumulative P&L vs. baselines")
    render_pnl_chart(pnl_df)

    with st.expander("Daily results"):
        st.dataframe(raw, use_container_width=True)


# --------------------------------------------------------------------------- #
# Render helpers
# --------------------------------------------------------------------------- #

def render_lmp_chart(lmp_df: pd.DataFrame, forecast_df: pd.DataFrame | None = None) -> None:
    """Render an interactive Plotly LMP time series chart.

    Args:
        lmp_df: DataFrame with columns [time, lmp] for the selected node.
        forecast_df: Optional forecast DataFrame [time, da_lmp_forecast] to overlay.
    """
    if lmp_df is None or lmp_df.empty:
        st.info("No LMP data for this node/date range.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lmp_df["time"], y=lmp_df["lmp"], name="Actual LMP", mode="lines"))
    if forecast_df is not None and not forecast_df.empty:
        ycol = "da_lmp_forecast" if "da_lmp_forecast" in forecast_df.columns else forecast_df.columns[-1]
        fig.add_trace(go.Scatter(x=forecast_df["time"], y=forecast_df[ycol], name="DA Forecast", mode="lines"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="$/MWh")
    st.plotly_chart(fig, use_container_width=True)


def render_soc_gauge(current_soc_mwh: float, capacity_mwh: float = 4.0) -> None:
    """Render a Plotly gauge showing current battery state-of-charge.

    Args:
        current_soc_mwh: Current SoC in MWh.
        capacity_mwh: Total battery capacity in MWh.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=current_soc_mwh,
            number={"suffix": " MWh"},
            gauge={
                "axis": {"range": [0, capacity_mwh]},
                "bar": {"color": "#2c7fb8"},
                "steps": [
                    {"range": [0, DEFAULT_BATTERY.soc_min_mwh], "color": "#fde0dd"},
                    {"range": [DEFAULT_BATTERY.soc_max_mwh, capacity_mwh], "color": "#fde0dd"},
                ],
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_dispatch_chart(schedule_df: pd.DataFrame) -> None:
    """Render a bar chart of the planned charge/discharge schedule.

    Args:
        schedule_df: DataFrame with columns [time, charge_mw, discharge_mw].
    """
    if schedule_df is None or schedule_df.empty:
        st.info("No dispatch schedule available.")
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(x=schedule_df["time"], y=schedule_df["discharge_mw"], name="Discharge", marker_color="#2ca02c"))
    # Charge shown below the zero line as negative power.
    fig.add_trace(go.Bar(x=schedule_df["time"], y=-schedule_df["charge_mw"], name="Charge", marker_color="#d62728"))
    fig.update_layout(barmode="relative", height=320, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="MW")
    st.plotly_chart(fig, use_container_width=True)


def render_pnl_chart(pnl_df: pd.DataFrame) -> None:
    """Render cumulative P&L vs. naive and perfect-hindsight baselines.

    Args:
        pnl_df: DataFrame with columns [date, our_revenue, naive_revenue, perfect_revenue].
    """
    if pnl_df is None or pnl_df.empty:
        st.info("No P&L data.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pnl_df["date"], y=pnl_df["our_revenue"].cumsum(), name="Our system"))
    fig.add_trace(go.Scatter(x=pnl_df["date"], y=pnl_df["naive_revenue"].cumsum(), name="Naive valley-fill"))
    fig.add_trace(go.Scatter(x=pnl_df["date"], y=pnl_df["perfect_revenue"].cumsum(), name="Perfect hindsight"))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Cumulative $")
    st.plotly_chart(fig, use_container_width=True)


def render_risk_event_log(events: pd.DataFrame | list[dict]) -> None:
    """Display a table of recent risk events / agent decisions.

    Args:
        events: DataFrame (or list of dicts) from the DuckDB agent_decisions table.
    """
    df = pd.DataFrame(events) if not isinstance(events, pd.DataFrame) else events
    if df.empty:
        st.info("No agent decisions logged yet.")
        return
    cols = [c for c in ["time", "agent", "action", "rationale"] if c in df.columns]
    st.dataframe(df[cols].sort_values("time", ascending=False).head(100), use_container_width=True)


def render_backtest_summary(pnl_df: pd.DataFrame) -> None:
    """Render a statistics panel: total P&L, Sharpe, max drawdown, win rate.

    Args:
        pnl_df: Daily P&L DataFrame with [date, our_revenue, naive_revenue, perfect_revenue].
    """
    if pnl_df is None or pnl_df.empty:
        st.info("No backtest data.")
        return

    rev = pnl_df["our_revenue"].to_numpy(dtype=float)
    naive = pnl_df["naive_revenue"].to_numpy(dtype=float)
    cumulative = np.cumsum(rev)

    total = float(rev.sum())
    total_naive = float(naive.sum())
    vs_naive_pct = ((total - total_naive) / abs(total_naive) * 100.0) if total_naive != 0 else float("nan")
    sharpe = (rev.mean() / rev.std() * np.sqrt(252)) if rev.std() > 0 else 0.0
    running_max = np.maximum.accumulate(cumulative)
    max_drawdown = float(np.max(running_max - cumulative)) if len(cumulative) else 0.0
    win_rate = float((rev > 0).mean() * 100.0)
    rmse = float(pnl_df["forecast_rmse"].mean()) if "forecast_rmse" in pnl_df.columns else float("nan")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total P&L", f"${total:,.0f}", f"{vs_naive_pct:+.1f}% vs naive")
    c2.metric("Sharpe (ann.)", f"{sharpe:.2f}")
    c3.metric("Max drawdown", f"${max_drawdown:,.0f}")
    c4.metric("Win rate", f"{win_rate:.0f}%")
    c5.metric("Forecast RMSE", f"${rmse:.1f}/MWh")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def _load_db(db_path: Path) -> Any | None:
    """Open a DuckDB-backed MarketDB for the dashboard.

    Args:
        db_path: Path to the DuckDB file.

    Returns:
        MarketDB instance, or None if the file does not exist.
    """
    if not Path(db_path).exists():
        return None
    from src.data.db import MarketDB

    return MarketDB(db_path)


def _load_recent_lmp(db: Any, node: str, days: int = 7) -> pd.DataFrame:
    """Load the most recent ``days`` of RT 5-min LMP for ``node``."""
    latest = db.get_latest_data_date("lmp")
    if latest is None:
        return pd.DataFrame(columns=["time", "lmp"])
    end = date.fromisoformat(latest)
    start = end - timedelta(days=days)
    df = db.query_lmp(node, "REAL_TIME_5_MIN", str(start), str(end))
    if df.empty:  # fall back to DA hourly if RT not present
        df = db.query_lmp(node, "DAY_AHEAD_HOURLY", str(start), str(end))
    return df


def _load_agent_decisions(db: Any) -> pd.DataFrame:
    """Read the agent_decisions audit table into a DataFrame."""
    try:
        return db._conn.execute(
            "SELECT time, agent, action, rationale, metadata FROM agent_decisions ORDER BY time DESC LIMIT 200"
        ).df()
    except Exception:
        return pd.DataFrame(columns=["time", "agent", "action", "rationale", "metadata"])


def _latest_soc_mwh(decisions: pd.DataFrame) -> float:
    """Extract the most recent battery SoC from optimizer decision metadata."""
    default = DEFAULT_BATTERY.initial_soc_pct * DEFAULT_BATTERY.capacity_mwh
    if decisions is None or decisions.empty or "metadata" not in decisions.columns:
        return default
    for _, row in decisions.iterrows():
        meta = row.get("metadata")
        if not meta:
            continue
        try:
            parsed = json.loads(meta) if isinstance(meta, str) else meta
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and "final_soc_mwh" in parsed:
            return float(parsed["final_soc_mwh"])
    return default


def _normalize_backtest_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a backtest results CSV to the [date, our/naive/perfect_revenue] schema."""
    df = pd.DataFrame()
    df["date"] = pd.to_datetime(raw["date"]) if "date" in raw.columns else range(len(raw))
    df["our_revenue"] = raw.get("revenue_usd", 0.0)
    df["naive_revenue"] = raw.get("naive_revenue_usd", 0.0)
    df["perfect_revenue"] = raw.get("perfect_revenue_usd", 0.0)
    if "forecast_rmse" in raw.columns:
        df["forecast_rmse"] = raw["forecast_rmse"]
    return df


if __name__ == "__main__":
    main()
