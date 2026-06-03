"""
Streamlit dashboard for the CAISO Battery Storage Trading Desk.

Editorial "almanac" presentation: warm-paper background, serif headlines,
monospace data labels, a single sage-green accent, hairline rules, and
square-grid infographics. Displays real-time and backtested system state:

    - Live LMP chart for NP15/SP15/ZP26 hubs (auto-refreshing every 5 min)
    - Battery SoC gauge and charge/discharge bar chart
    - Forecasted vs. actual LMP overlay
    - Daily / cumulative P&L vs. baseline strategies
    - Risk event log with timestamps and rationale
    - Agent decision audit trail
    - Backtest summary statistics (RMSE, Sharpe, max drawdown)

Run with:
    streamlit run src/dashboard/app.py

Cumulative-P&L and other backtest charts are rendered server-side as static
images (matplotlib) for pixel-precise editorial styling and to avoid stale
client-side chart caching.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import matplotlib
matplotlib.use("Agg")  # headless backend; no display required
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.config.battery import DEFAULT_BATTERY
from src.config.nodes import CAISO_HUB_NODES, DEFAULT_HUB_NODE, node_short_name

# DuckDB path (override via the sidebar)
_DEFAULT_DB_PATH: Path = Path("data/market.duckdb")

# Default backtest results CSV produced by `python main.py backtest`
_DEFAULT_BACKTEST_CSV: Path = Path("results/backtest_2024H2.csv")

# The hub node the default reference backtest CSV was generated on. Per-node
# artifacts (if produced) follow the `backtest_2024H2_<SHORT>.csv` convention.
_BACKTEST_REFERENCE_NODE: str = "TH_NP15_GEN-APND"

# Auto-refresh interval in seconds (matches CAISO 5-min RT dispatch)
_REFRESH_INTERVAL_S: int = 300

# --------------------------------------------------------------------------- #
# Theme — AI-almanac editorial palette
# --------------------------------------------------------------------------- #

_COLORS = {
    "paper": "#fbfaf2",      # warm off-white background
    "paper2": "#f3f1e6",     # faint panel tint
    "ink": "#1b1b18",        # near-black text
    "ink_soft": "#6c6a5f",   # muted text / labels
    "rule": "#dcd8c8",       # hairline rules / borders
    "grid": "#e7e4d6",       # chart gridlines
    "green": "#5f7257",      # sage accent (positive)
    "green_dk": "#45543e",   # deep sage
    "green_lt": "#aab59c",   # pale sage
    "charcoal": "#3a3a34",   # secondary line (perfect hindsight)
    "brick": "#9b4a3f",      # muted brick red (loss / negative)
    "gold": "#9c8456",       # tertiary accent
}

# matplotlib defaults for the editorial look
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Charter", "Georgia", "Palatino", "DejaVu Serif"],
    "font.monospace": ["Andale Mono", "Courier New", "Menlo", "DejaVu Sans Mono"],
    "axes.edgecolor": _COLORS["rule"],
    "axes.linewidth": 0.8,
    "text.color": _COLORS["ink"],
    "axes.labelcolor": _COLORS["ink_soft"],
    "xtick.color": _COLORS["ink_soft"],
    "ytick.color": _COLORS["ink_soft"],
    "svg.fonttype": "none",
})

# Plotly template for the (live-page) interactive charts, light editorial theme.
_PLOT_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="'EB Garamond', Georgia, serif", color=_COLORS["ink"], size=14),
        paper_bgcolor=_COLORS["paper"],
        plot_bgcolor=_COLORS["paper"],
        colorway=[_COLORS["green"], _COLORS["charcoal"], _COLORS["gold"], _COLORS["brick"]],
        margin=dict(l=12, r=12, t=30, b=12),
        hoverlabel=dict(bgcolor=_COLORS["paper2"], bordercolor=_COLORS["rule"],
                        font=dict(color=_COLORS["ink"], size=12,
                                  family="'Space Mono', monospace")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(color=_COLORS["ink_soft"], family="'Space Mono', monospace", size=11)),
        xaxis=dict(gridcolor=_COLORS["grid"], zerolinecolor=_COLORS["rule"], linecolor=_COLORS["rule"],
                   tickfont=dict(color=_COLORS["ink_soft"], family="'Space Mono', monospace", size=11)),
        yaxis=dict(gridcolor=_COLORS["grid"], zerolinecolor=_COLORS["rule"], linecolor=_COLORS["rule"],
                   tickfont=dict(color=_COLORS["ink_soft"], family="'Space Mono', monospace", size=11)),
    )
)


def _plotly(fig: go.Figure, height: int = 360) -> None:
    """Apply the editorial template and render a Plotly figure."""
    fig.update_layout(template=_PLOT_TEMPLATE, height=height)
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# matplotlib static-chart plumbing
# --------------------------------------------------------------------------- #

def _new_axes(height_px: int) -> tuple[plt.Figure, plt.Axes]:
    """Create a themed matplotlib figure/axes on warm paper."""
    dpi = 100.0
    fig, ax = plt.subplots(figsize=(13.2, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor(_COLORS["paper"])
    ax.set_facecolor(_COLORS["paper"])
    ax.grid(True, axis="y", color=_COLORS["grid"], lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_COLORS["rule"])
    ax.spines["bottom"].set_color(_COLORS["rule"])
    ax.tick_params(length=0, labelsize=10.5)
    return fig, ax


def _mono_ticks(ax: plt.Axes) -> None:
    """Render tick labels in the monospace 'typewriter' face."""
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("monospace")
        lbl.set_fontsize(10)


def _png(fig: plt.Figure) -> bytes:
    """Serialize a matplotlib figure to PNG bytes and close it."""
    buf = BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #

def _inject_css() -> None:
    """Inject the editorial light theme."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=Space+Mono:wght@400;700&display=swap');

        :root {{
            --paper: {_COLORS["paper"]}; --ink: {_COLORS["ink"]}; --soft: {_COLORS["ink_soft"]};
            --rule: {_COLORS["rule"]}; --green: {_COLORS["green"]}; --brick: {_COLORS["brick"]};
        }}

        .stApp {{ background: {_COLORS["paper"]}; color: {_COLORS["ink"]}; }}
        html, body, [class*="css"] {{ font-family: 'EB Garamond', Georgia, serif; }}
        .block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1480px; }}
        #MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}

        /* Mono helper */
        .mono {{ font-family: 'Space Mono', monospace; }}

        /* ---- Masthead ---- */
        .mast {{ border-top: 2px solid {_COLORS["ink"]}; padding-top: 14px; margin-bottom: 6px; }}
        .mast-row {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; }}
        .mast-title {{ font-size: 2.35rem; font-weight: 600; line-height: 1.05; letter-spacing: -0.01em;
                       margin: 0; color: {_COLORS["ink"]}; }}
        .mast-title .bolt {{ color: {_COLORS["green"]}; }}
        .mast-badge {{ font-family: 'Space Mono', monospace; font-size: 0.72rem; font-weight: 700;
                       letter-spacing: 0.08em; color: {_COLORS["paper"]}; background: {_COLORS["ink"]};
                       padding: 6px 12px; border-radius: 3px; white-space: nowrap; }}
        .mast-badge.live {{ background: {_COLORS["green"]}; }}
        .mast-badge.live::before {{ content: "● "; }}
        .mast-sub {{ font-family: 'Space Mono', monospace; font-size: 0.78rem; letter-spacing: 0.02em;
                     color: {_COLORS["ink_soft"]}; margin-top: 10px; text-transform: uppercase; }}
        .mast-rule {{ border-bottom: 1px solid {_COLORS["rule"]}; margin: 12px 0 26px; }}

        /* ---- Section headers ---- */
        .sec {{ margin: 34px 0 14px; border-bottom: 1px solid {_COLORS["rule"]}; padding-bottom: 7px; }}
        .sec-kicker {{ font-family: 'Space Mono', monospace; font-size: 0.68rem; font-weight: 700;
                       letter-spacing: 0.16em; text-transform: uppercase; color: {_COLORS["green"]}; }}
        .sec-title {{ font-size: 1.45rem; font-weight: 600; color: {_COLORS["ink"]}; margin: 3px 0 0;
                      line-height: 1.1; }}

        /* ---- KPI almanac row ---- */
        .kpi-row {{ display: grid; grid-template-columns: repeat(6, 1fr); margin: 6px 0 8px; }}
        .kpi-cell {{ padding: 4px 18px; border-left: 1px solid {_COLORS["rule"]}; }}
        .kpi-cell:first-child {{ border-left: none; padding-left: 2px; }}
        .kpi-k {{ font-family: 'Space Mono', monospace; font-size: 0.64rem; font-weight: 700;
                  letter-spacing: 0.11em; text-transform: uppercase; color: {_COLORS["ink_soft"]}; }}
        .kpi-v {{ font-family: 'Space Mono', monospace; font-size: 1.62rem; font-weight: 700;
                  color: {_COLORS["ink"]}; line-height: 1.15; margin-top: 7px; }}
        .kpi-v.pos {{ color: {_COLORS["green_dk"]}; }}
        .kpi-v.neg {{ color: {_COLORS["brick"]}; }}
        .kpi-s {{ font-family: 'Space Mono', monospace; font-size: 0.70rem; color: {_COLORS["ink_soft"]};
                  margin-top: 5px; }}
        .kpi-s.pos {{ color: {_COLORS["green"]}; }}
        .kpi-s.neg {{ color: {_COLORS["brick"]}; }}

        /* ---- Dot-grid almanac (trading calendar) ---- */
        .alm {{ display: flex; gap: 30px; align-items: flex-start; flex-wrap: wrap; margin-top: 6px; }}
        .dotgrid {{ display: grid; grid-template-columns: repeat(32, 1fr); gap: 3px; flex: 1 1 620px; }}
        .dot {{ width: 100%; aspect-ratio: 1 / 1; border-radius: 2px; }}
        .dot.profit {{ background: {_COLORS["green"]}; }}
        .dot.loss {{ background: {_COLORS["brick"]}; }}
        .dot.flat {{ background: transparent; border: 1px solid {_COLORS["rule"]}; }}
        .alm-key {{ flex: 0 0 190px; }}
        .alm-key .row {{ display: flex; align-items: center; gap: 9px; font-family: 'Space Mono', monospace;
                         font-size: 0.74rem; color: {_COLORS["ink_soft"]}; margin-bottom: 9px; }}
        .alm-key .sw {{ width: 13px; height: 13px; border-radius: 2px; display: inline-block; }}
        .alm-key .sw.profit {{ background: {_COLORS["green"]}; }}
        .alm-key .sw.loss {{ background: {_COLORS["brick"]}; }}
        .alm-key .sw.flat {{ background: transparent; border: 1px solid {_COLORS["rule"]}; }}
        .alm-key b {{ color: {_COLORS["ink"]}; }}

        /* Images (static charts) blend into paper */
        [data-testid="stImage"] img {{ border-radius: 2px; }}

        /* Captions */
        [data-testid="stCaptionContainer"], .stCaption {{ font-family: 'EB Garamond', serif !important;
            font-style: italic; color: {_COLORS["ink_soft"]} !important; }}

        /* Sidebar */
        [data-testid="stSidebar"] {{ background: {_COLORS["paper2"]}; border-right: 1px solid {_COLORS["rule"]}; }}
        [data-testid="stSidebar"] * {{ color: {_COLORS["ink"]}; }}

        /* Dataframe */
        [data-testid="stDataFrame"] {{ border: 1px solid {_COLORS["rule"]}; border-radius: 3px; }}

        /* Expander */
        [data-testid="stExpander"] {{ border: 1px solid {_COLORS["rule"]} !important; border-radius: 3px; }}

        /* Radio / widgets accent */
        [data-baseweb="radio"] [aria-checked="true"] div {{ background-color: {_COLORS["green"]} !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Layout primitives
# --------------------------------------------------------------------------- #

def _masthead(subtitle: str, badge_text: str, live: bool = False) -> None:
    """Render the editorial masthead band."""
    cls = "mast-badge live" if live else "mast-badge"
    st.markdown(
        f"""
        <div class="mast">
          <div class="mast-row">
            <h1 class="mast-title">CAISO Battery Storage Trading Desk</h1>
            <span class="{cls}">{badge_text}</span>
          </div>
          <div class="mast-sub">{subtitle}</div>
        </div>
        <div class="mast-rule"></div>
        """,
        unsafe_allow_html=True,
    )


def _section(title: str, kicker: str = "") -> None:
    """Render a section header with a monospace kicker and serif title."""
    kick = f'<div class="sec-kicker">{kicker}</div>' if kicker else ""
    st.markdown(f'<div class="sec">{kick}<div class="sec-title">{title}</div></div>',
                unsafe_allow_html=True)


def main() -> None:
    """Entry point for the Streamlit dashboard."""
    st.set_page_config(page_title="CAISO Battery Trading Desk", page_icon="⚡",
                       layout="wide", initial_sidebar_state="expanded")
    _inject_css()
    config = render_sidebar()
    if config["mode"] == "backtest":
        _render_backtest_page(config)
    else:
        _render_live_page(config)


def render_sidebar() -> dict[str, Any]:
    """Render the sidebar with mode selection and configuration."""
    st.sidebar.markdown(
        "<div class='mono' style='font-size:0.72rem;letter-spacing:0.14em;"
        "text-transform:uppercase;color:#5f7257;font-weight:700'>The Desk</div>"
        "<div style='font-size:1.25rem;font-weight:600;margin-bottom:14px'>Control Panel</div>",
        unsafe_allow_html=True,
    )
    mode_label = st.sidebar.radio("Mode", ["Backtest", "Live / Paper-trade"])
    mode = "backtest" if mode_label == "Backtest" else "live"

    node = st.sidebar.selectbox(
        "Hub node",
        CAISO_HUB_NODES,
        index=CAISO_HUB_NODES.index(DEFAULT_HUB_NODE) if DEFAULT_HUB_NODE in CAISO_HUB_NODES else 0,
        format_func=lambda n: f"{node_short_name(n)}  ·  {n}",
    )
    with st.sidebar.expander("Data sources", expanded=False):
        db_path = st.text_input("DuckDB path", value=str(_DEFAULT_DB_PATH))
        backtest_csv = st.text_input("Backtest CSV", value=str(_DEFAULT_BACKTEST_CSV))

    b = DEFAULT_BATTERY
    st.sidebar.markdown(
        f"""
        <div style="border-top:1px solid {_COLORS['rule']};margin-top:14px;padding-top:12px;
                    font-family:'Space Mono',monospace;font-size:0.72rem;color:{_COLORS['ink_soft']};
                    line-height:1.9">
        <div style="font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
                    color:{_COLORS['green']};margin-bottom:6px">BESS Specification</div>
        Capacity&nbsp;&nbsp;<b style="color:{_COLORS['ink']};float:right">{b.capacity_mwh:.0f} MWh</b><br>
        Power&nbsp;&nbsp;<b style="color:{_COLORS['ink']};float:right">{b.max_discharge_mw:.0f} MW</b><br>
        Round-trip&nbsp;η&nbsp;&nbsp;<b style="color:{_COLORS['ink']};float:right">{b.round_trip_efficiency*100:.0f}%</b><br>
        SoC&nbsp;band&nbsp;&nbsp;<b style="color:{_COLORS['ink']};float:right">{b.soc_min_mwh:.1f}–{b.soc_max_mwh:.1f} MWh</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if mode == "live":
        st.sidebar.caption(f"Auto-refreshing every {_REFRESH_INTERVAL_S // 60} min.")
        st.markdown(f'<meta http-equiv="refresh" content="{_REFRESH_INTERVAL_S}">',
                    unsafe_allow_html=True)

    return {"mode": mode, "node": node, "db_path": Path(db_path), "backtest_csv": Path(backtest_csv)}


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #

def _render_live_page(config: dict[str, Any]) -> None:
    """Render the live / paper-trade page from DuckDB state."""
    node = config["node"]
    _masthead(
        subtitle=f"Live paper-trade · {node_short_name(node)} hub · real-time CAISO 5-minute dispatch",
        badge_text=f"LIVE · {node_short_name(node)}", live=True,
    )

    db = _load_db(config["db_path"])
    if db is None:
        st.warning(f"No DuckDB database found at `{config['db_path']}`. Run `main.py data-fetch` first.")
        return

    decisions = _load_agent_decisions(db)
    current_soc = _latest_soc_mwh(decisions)

    col1, col2 = st.columns([3, 1])
    with col1:
        _section(f"Recent LMP — {node_short_name(node)}", "Locational Marginal Price")
        render_lmp_chart(_load_recent_lmp(db, node))
    with col2:
        _section("Battery SoC", "State of Charge")
        render_soc_gauge(current_soc, DEFAULT_BATTERY.capacity_mwh)

    _section("Risk Events & Agent Decisions", "Audit Trail")
    render_risk_event_log(decisions)


def _render_backtest_page(config: dict[str, Any]) -> None:
    """Render the backtest page from a results CSV."""
    csv_path, data_node, is_fallback = _resolve_backtest_csv(config["backtest_csv"], config["node"])
    if csv_path is None:
        _masthead(subtitle="No backtest artifact found", badge_text="BACKTEST")
        st.warning(
            f"No backtest results at `{config['backtest_csv']}`. "
            f"Run `python main.py backtest --output {config['backtest_csv']}` first."
        )
        return

    raw = pd.read_csv(csv_path)
    pnl_df = _normalize_backtest_df(raw)

    period = ""
    if "date" in raw.columns and len(raw):
        d0, d1 = pd.to_datetime(raw["date"]).min(), pd.to_datetime(raw["date"]).max()
        period = f"{d0:%b %d, %Y} – {d1:%b %d, %Y} · {len(raw)} trading days"

    _masthead(
        subtitle=f"Out-of-sample backtest · {node_short_name(data_node)} hub · {period}",
        badge_text=f"BACKTEST · {node_short_name(data_node)}",
    )
    if is_fallback:
        st.info(
            f"No dedicated backtest artifact for **{node_short_name(config['node'])}** "
            f"(`{config['node']}`). Showing the **{node_short_name(data_node)}** reference run. "
            f"Generate per-node results with `python main.py backtest --node {config['node']} "
            f"--output results/backtest_2024H2_{node_short_name(config['node'])}.csv`.",
            icon="ℹ️",
        )

    _section("Performance Summary", "2024 H2 · Out-of-Sample")
    render_backtest_summary(pnl_df, raw)

    _section("Cumulative P&L vs. Baseline Strategies", "Equity Curves")
    st.image(_pnl_png(pnl_df, include_naive=True, height_px=430), use_column_width=True)
    st.caption(
        "The naive valley-fill strategy bleeds capital on 2024 H2 volatility — note the deep negative "
        "plunge — while the agent system holds net-positive. Agent and perfect-hindsight curves hug the "
        "zero line at this scale; the detail panel below zooms in on that gap."
    )

    _section("Agent vs. Perfect Hindsight", "Detail · Same Axis Zoom")
    st.image(_pnl_png(pnl_df, include_naive=False, height_px=300), use_column_width=True)

    _section("Trading Activity", f"{len(raw)} Sessions · One Square Per Day")
    render_trading_calendar(pnl_df, raw)

    col_a, col_b = st.columns(2)
    with col_a:
        _section("Daily P&L", "Per-Session Realized")
        st.image(_daily_pnl_png(pnl_df), use_column_width=True)
    with col_b:
        _section("Forecast Accuracy", "RMSE per Session")
        st.image(_rmse_png(pnl_df), use_column_width=True)

    if "cycles" in raw.columns:
        _section("Battery Utilization", "Equivalent Cycles per Day")
        st.image(_cycles_png(raw), use_column_width=True)

    with st.expander("Daily results (raw)"):
        st.dataframe(_format_results_table(raw), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Static (matplotlib) chart renderers
# --------------------------------------------------------------------------- #

def _pnl_png(pnl_df: pd.DataFrame, include_naive: bool, height_px: int = 430) -> bytes:
    """Cumulative-P&L line chart rendered to a static editorial PNG.

    Args:
        pnl_df: DataFrame with [date, our_revenue, perfect_revenue] and, when
            ``include_naive`` is True, ``naive_revenue``.
        include_naive: Whether to draw the naive valley-fill baseline.
        height_px: Target image height in pixels.

    Returns:
        PNG-encoded image bytes.
    """
    x = pd.to_datetime(pnl_df["date"])
    our = pnl_df["our_revenue"].cumsum()
    perfect = pnl_df["perfect_revenue"].cumsum()

    fig, ax = _new_axes(height_px)
    ax.plot(x, perfect, color=_COLORS["charcoal"], lw=1.6, ls=(0, (5, 3)), label="Perfect hindsight")
    ax.fill_between(x, our, 0, color=_COLORS["green"], alpha=0.14, lw=0)
    ax.plot(x, our, color=_COLORS["green"], lw=2.6, label="Agent system")
    if include_naive:
        naive = pnl_df["naive_revenue"].cumsum()
        ax.fill_between(x, naive, 0, color=_COLORS["brick"], alpha=0.10, lw=0)
        ax.plot(x, naive, color=_COLORS["brick"], lw=2.4, label="Naive valley-fill")
    ax.axhline(0, color=_COLORS["ink_soft"], lw=0.9, ls=":")

    ax.set_ylabel("cumulative $", fontsize=11, style="italic")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    _mono_ticks(ax)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=10.5,
                    ncol=3, handlelength=1.6, columnspacing=1.8)
    for t in leg.get_texts():
        t.set_color(_COLORS["ink"])
        t.set_fontfamily("monospace")
        t.set_fontsize(9.5)
    return _png(fig)


def _daily_pnl_png(pnl_df: pd.DataFrame) -> bytes:
    """Per-day realized P&L as green/brick bars."""
    x = pd.to_datetime(pnl_df["date"])
    rev = pnl_df["our_revenue"]
    colors = [_COLORS["green"] if v >= 0 else _COLORS["brick"] for v in rev]
    fig, ax = _new_axes(300)
    ax.bar(x, rev, color=colors, width=2.0)
    ax.axhline(0, color=_COLORS["ink_soft"], lw=0.9)
    ax.set_ylabel("$ / day", fontsize=11, style="italic")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    _mono_ticks(ax)
    return _png(fig)


def _rmse_png(pnl_df: pd.DataFrame) -> bytes:
    """Daily forecast RMSE with the mean as a reference line."""
    fig, ax = _new_axes(300)
    if "forecast_rmse" not in pnl_df.columns:
        ax.text(0.5, 0.5, "no forecast-RMSE data", ha="center", va="center",
                color=_COLORS["ink_soft"], family="monospace", transform=ax.transAxes)
        return _png(fig)
    x = pd.to_datetime(pnl_df["date"])
    rmse = pnl_df["forecast_rmse"]
    mean_rmse = float(rmse.mean())
    ax.fill_between(x, rmse, 0, color=_COLORS["green"], alpha=0.10, lw=0)
    ax.plot(x, rmse, color=_COLORS["green_dk"], lw=1.8)
    ax.axhline(mean_rmse, color=_COLORS["gold"], lw=1.3, ls="--")
    ax.text(x.iloc[2], mean_rmse, f"  mean ${mean_rmse:.1f}", color=_COLORS["gold"],
            family="monospace", fontsize=9, va="bottom")
    ax.set_ylabel("$/MWh", fontsize=11, style="italic")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    _mono_ticks(ax)
    return _png(fig)


def _cycles_png(raw: pd.DataFrame) -> bytes:
    """Per-day equivalent battery cycles with the daily limit marked."""
    x = pd.to_datetime(raw["date"]) if "date" in raw.columns else pd.RangeIndex(len(raw))
    fig, ax = _new_axes(280)
    ax.bar(x, raw["cycles"], color=_COLORS["green"], width=2.0)
    limit = DEFAULT_BATTERY.max_cycles_per_day
    ax.axhline(limit, color=_COLORS["brick"], lw=1.2, ls=":")
    ax.text(x.iloc[2] if hasattr(x, "iloc") else 2, limit, f"  limit {limit:.0f}/day",
            color=_COLORS["brick"], family="monospace", fontsize=9, va="bottom")
    ax.set_ylabel("equiv. cycles", fontsize=11, style="italic")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    _mono_ticks(ax)
    return _png(fig)


# --------------------------------------------------------------------------- #
# Live-page interactive (Plotly) charts
# --------------------------------------------------------------------------- #

def render_lmp_chart(lmp_df: pd.DataFrame, forecast_df: pd.DataFrame | None = None) -> None:
    """Render an interactive LMP time series chart."""
    if lmp_df is None or lmp_df.empty:
        st.info("No LMP data for this node/date range.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lmp_df["time"], y=lmp_df["lmp"], name="Actual LMP", mode="lines",
                             line=dict(color=_COLORS["green"], width=1.8),
                             fill="tozeroy", fillcolor="rgba(95,114,87,0.10)"))
    if forecast_df is not None and not forecast_df.empty:
        ycol = "da_lmp_forecast" if "da_lmp_forecast" in forecast_df.columns else forecast_df.columns[-1]
        fig.add_trace(go.Scatter(x=forecast_df["time"], y=forecast_df[ycol], name="DA Forecast",
                                 mode="lines", line=dict(color=_COLORS["gold"], width=1.6, dash="dot")))
    fig.update_layout(yaxis_title="$/MWh")
    _plotly(fig, height=380)


def render_soc_gauge(current_soc_mwh: float, capacity_mwh: float = 4.0) -> None:
    """Render a gauge of current battery state-of-charge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_soc_mwh,
        number={"suffix": " MWh", "font": {"color": _COLORS["ink"], "size": 28,
                                           "family": "'Space Mono', monospace"}},
        gauge={
            "axis": {"range": [0, capacity_mwh], "tickcolor": _COLORS["ink_soft"]},
            "bar": {"color": _COLORS["green"]},
            "bgcolor": _COLORS["paper2"],
            "borderwidth": 1, "bordercolor": _COLORS["rule"],
            "steps": [
                {"range": [0, DEFAULT_BATTERY.soc_min_mwh], "color": "rgba(155,74,63,0.18)"},
                {"range": [DEFAULT_BATTERY.soc_max_mwh, capacity_mwh], "color": "rgba(155,74,63,0.18)"},
            ],
        },
    ))
    _plotly(fig, height=300)


def render_dispatch_chart(schedule_df: pd.DataFrame) -> None:
    """Render a bar chart of the planned charge/discharge schedule."""
    if schedule_df is None or schedule_df.empty:
        st.info("No dispatch schedule available.")
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(x=schedule_df["time"], y=schedule_df["discharge_mw"],
                         name="Discharge", marker_color=_COLORS["green"]))
    fig.add_trace(go.Bar(x=schedule_df["time"], y=-schedule_df["charge_mw"],
                         name="Charge", marker_color=_COLORS["brick"]))
    fig.update_layout(barmode="relative", yaxis_title="MW")
    _plotly(fig, height=320)


def render_risk_event_log(events: pd.DataFrame | list[dict]) -> None:
    """Display a table of recent risk events / agent decisions."""
    df = pd.DataFrame(events) if not isinstance(events, pd.DataFrame) else events
    if df.empty:
        st.info("No agent decisions logged yet.")
        return
    cols = [c for c in ["time", "agent", "action", "rationale"] if c in df.columns]
    st.dataframe(df[cols].sort_values("time", ascending=False).head(100),
                 use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# KPI row + dot-grid almanac
# --------------------------------------------------------------------------- #

def render_backtest_summary(pnl_df: pd.DataFrame, raw: pd.DataFrame | None = None) -> None:
    """Render the editorial KPI almanac row."""
    if pnl_df is None or pnl_df.empty:
        st.info("No backtest data.")
        return

    rev = pnl_df["our_revenue"].to_numpy(dtype=float)
    naive = pnl_df["naive_revenue"].to_numpy(dtype=float)
    perfect = pnl_df["perfect_revenue"].to_numpy(dtype=float)
    cumulative = np.cumsum(rev)

    total = float(rev.sum())
    total_naive = float(naive.sum())
    total_perfect = float(perfect.sum())
    edge_vs_naive = total - total_naive
    sharpe = (rev.mean() / rev.std() * np.sqrt(252)) if rev.std() > 0 else 0.0
    running_max = np.maximum.accumulate(cumulative)
    max_drawdown = float(np.max(running_max - cumulative)) if len(cumulative) else 0.0
    capture = (total / total_perfect * 100.0) if total_perfect > 0 else float("nan")
    rmse = float(pnl_df["forecast_rmse"].mean()) if "forecast_rmse" in pnl_df.columns else float("nan")

    n_days = len(rev)
    days_beat_naive = int((rev - naive > 0).sum())
    beat_pct = days_beat_naive / n_days * 100.0 if n_days else 0.0
    active = int((raw["cycles"].to_numpy(dtype=float) > 0).sum()) if (
        raw is not None and "cycles" in raw.columns) else int((rev != 0).sum())
    profitable = int((rev > 0).sum())

    cells = [
        ("Net P&L", f"${total:,.0f}", "pos", f"+${edge_vs_naive:,.0f} vs naive", "pos"),
        ("Edge vs Naive", f"+${edge_vs_naive:,.0f}", "pos", f"naive lost ${total_naive:,.0f}", "neg"),
        ("Days Beat Naive", f"{beat_pct:.0f}%", "pos", f"{days_beat_naive}/{n_days} days", ""),
        ("Sharpe (ann.)", f"{sharpe:.2f}", "", "risk-adjusted", ""),
        ("Max Drawdown", f"${max_drawdown:,.0f}", "neg", f"{capture:.0f}% of perfect", ""),
        ("Forecast RMSE", f"${rmse:.1f}", "", f"{active} active · {profitable} profit", ""),
    ]
    html = '<div class="kpi-row">'
    for k, v, vtone, s, stone in cells:
        html += (
            f'<div class="kpi-cell"><div class="kpi-k">{k}</div>'
            f'<div class="kpi-v {vtone}">{v}</div>'
            f'<div class="kpi-s {stone}">{s}</div></div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_trading_calendar(pnl_df: pd.DataFrame, raw: pd.DataFrame | None = None) -> None:
    """Render a one-square-per-day almanac grid colored by daily outcome."""
    rev = pnl_df["our_revenue"].to_numpy(dtype=float)
    n = len(rev)
    n_profit = int((rev > 0).sum())
    n_loss = int((rev < 0).sum())
    n_flat = n - n_profit - n_loss

    dots = []
    for v in rev:
        cls = "profit" if v > 0 else ("loss" if v < 0 else "flat")
        dots.append(f'<div class="dot {cls}"></div>')
    grid = "".join(dots)

    st.markdown(
        f"""
        <div class="alm">
          <div class="dotgrid">{grid}</div>
          <div class="alm-key">
            <div class="row"><span class="sw profit"></span> <b>{n_profit}</b>&nbsp;profitable</div>
            <div class="row"><span class="sw loss"></span> <b>{n_loss}</b>&nbsp;loss days</div>
            <div class="row"><span class="sw flat"></span> <b>{n_flat}</b>&nbsp;flat (held)</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "The optimizer holds flat on most sessions — trading only when the forecast price spread clears "
        "round-trip efficiency and degradation cost — which is why the equity curve stays disciplined."
    )


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def _load_db(db_path: Path) -> Any | None:
    """Open a DuckDB-backed MarketDB for the dashboard."""
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
    if df.empty:
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


def _resolve_backtest_csv(base_csv: Path, node: str) -> tuple[Path | None, str, bool]:
    """Resolve which backtest CSV to display for the selected node.

    Prefers a per-node artifact (``<base>_<SHORT>.csv``); falls back to the
    shared reference CSV when no node-specific file exists.

    Returns:
        ``(csv_path | None, data_node, is_fallback)``.
    """
    base = Path(base_csv)
    per_node = base.with_name(f"{base.stem}_{node_short_name(node)}{base.suffix}")
    if per_node.exists():
        return per_node, node, False
    if base.exists():
        return base, _BACKTEST_REFERENCE_NODE, node != _BACKTEST_REFERENCE_NODE
    return None, node, False


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


def _format_results_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a display-friendly copy of the raw results with tidy column names."""
    rename = {
        "date": "Date", "revenue_usd": "P&L ($)", "cycles": "Cycles",
        "final_soc_mwh": "Final SoC (MWh)", "vs_naive_usd": "vs Naive ($)",
        "vs_perfect_usd": "vs Perfect ($)", "forecast_rmse": "RMSE ($/MWh)",
        "solver_status": "Solver", "naive_revenue_usd": "Naive P&L ($)",
        "perfect_revenue_usd": "Perfect P&L ($)",
    }
    df = raw.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].round(2)
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


if __name__ == "__main__":
    main()
