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
and optionally to a live state file written by the coordinator during
paper-trading mode.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# DuckDB path (override via env var or Streamlit secrets)
_DEFAULT_DB_PATH: Path = Path("data/market.duckdb")

# Auto-refresh interval in seconds (matches CAISO 5-min RT dispatch)
_REFRESH_INTERVAL_S: int = 300


def main() -> None:
    """Entry point for the Streamlit dashboard."""
    raise NotImplementedError(
        # st.set_page_config(title='CAISO Battery Trading Desk', layout='wide')
        # Render sidebar, then route to selected page via st.sidebar.radio.
    )


def render_sidebar() -> dict[str, str | bool]:
    """Render the sidebar with mode selection and configuration.

    Returns:
        Dict with keys: 'mode' ('live' | 'backtest'), 'node', 'db_path'.
    """
    raise NotImplementedError(
        # st.sidebar.title('Settings')
        # mode = st.sidebar.radio('Mode', ['Live / Paper-trade', 'Backtest'])
        # node = st.sidebar.selectbox('Hub node', ['TH_NP15_GEN-APND', ...])
        # Return config dict.
    )


def render_lmp_chart(lmp_df: pd.DataFrame, forecast_df: pd.DataFrame | None = None) -> None:
    """Render an interactive Plotly LMP time series chart.

    Args:
        lmp_df: DataFrame with columns [time, lmp] for the selected node.
        forecast_df: Optional forecast DataFrame [time, da_lmp_forecast] to overlay.
    """
    raise NotImplementedError(
        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=lmp_df.time, y=lmp_df.lmp, name='Actual RT LMP'))
        # if forecast_df is not None:
        #   fig.add_trace(go.Scatter(..., name='DA Forecast'))
        # st.plotly_chart(fig, use_container_width=True)
    )


def render_soc_gauge(current_soc_mwh: float, capacity_mwh: float = 4.0) -> None:
    """Render a Plotly gauge showing current battery state-of-charge.

    Args:
        current_soc_mwh: Current SoC in MWh.
        capacity_mwh: Total battery capacity in MWh.
    """
    raise NotImplementedError(
        # fig = go.Figure(go.Indicator(mode='gauge+number', value=current_soc_mwh,
        #   gauge={'axis': {'range': [0, capacity_mwh]}, ...}))
        # st.plotly_chart(fig)
    )


def render_dispatch_chart(schedule_df: pd.DataFrame) -> None:
    """Render a bar chart of the planned charge/discharge schedule.

    Args:
        schedule_df: DataFrame with columns [time, charge_mw, discharge_mw].
    """
    raise NotImplementedError(
        # Stacked bar chart: discharge above zero-line, charge below (negative).
    )


def render_pnl_chart(pnl_df: pd.DataFrame) -> None:
    """Render cumulative P&L vs. naive and perfect-hindsight baselines.

    Args:
        pnl_df: DataFrame with columns [date, our_revenue, naive_revenue, perfect_revenue].
    """
    raise NotImplementedError(
        # Three line traces on one chart; fill area between our_revenue and naive.
    )


def render_risk_event_log(events: list[dict]) -> None:
    """Display a table of recent risk events with severity color coding.

    Args:
        events: List of risk event dicts from DuckDB agent_decisions table.
    """
    raise NotImplementedError(
        # st.dataframe with conditional formatting on 'level' column.
    )


def render_backtest_summary(pnl_df: pd.DataFrame) -> None:
    """Render a statistics panel: total P&L, Sharpe, max drawdown, win rate.

    Args:
        pnl_df: Daily P&L DataFrame.
    """
    raise NotImplementedError(
        # Compute metrics from pnl_df, display as st.metric() tiles.
        # Sharpe = mean(daily_return) / std(daily_return) * sqrt(252)
        # Max drawdown = max(cummax(cumulative) - cumulative)
        # Win rate = fraction of days with positive revenue
    )


def _load_db(db_path: Path) -> "Any":  # MarketDB
    """Open a read-only DuckDB connection for the dashboard.

    Args:
        db_path: Path to the DuckDB file.

    Returns:
        MarketDB instance in read-only mode.
    """
    raise NotImplementedError(
        # from src.data.db import MarketDB
        # return MarketDB(db_path)
    )


if __name__ == "__main__":
    main()
