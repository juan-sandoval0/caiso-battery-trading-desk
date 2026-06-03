"""
Wiring smoke test for the LangGraph coordinator.

Builds the compiled graph with the live data fetch, market-intel, and forecaster
nodes mocked (no network / API key / future-dated data needed) but the real
RiskMonitor, DispatchOptimizer, ConflictResolver, and finalize nodes, then runs
a single tick end-to-end against an in-memory DuckDB. Verifies the
fetch -> market_intel -> forecast -> risk -> optimize -> resolve -> finalize
path completes and produces a dispatch schedule.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.agents.optimizer import DispatchOptimizerAgent
from src.agents.risk_monitor import RiskMonitorAgent
from src.config.battery import DEFAULT_BATTERY
from src.config.nodes import DEFAULT_HUB_NODE
from src.data.db import MarketDB
from src.agents.forecaster import PriceForecast
from src.orchestrator.coordinator import build_graph, GraphState
from src.orchestrator.shared_state import SharedMarketState


def _mock_forecaster(spread: bool = True) -> MagicMock:
    """Forecaster whose run() returns a 24h PriceForecast with a price spread."""
    prices = np.full(24, 20.0)
    if spread:
        prices[18:21] = 300.0
    fc = PriceForecast(
        node=DEFAULT_HUB_NODE,
        times=pd.date_range("2024-07-01", periods=24, freq="h"),
        da_lmp_forecast=prices,
        rt_lmp_forecast=None,
        confidence_low=None,
        confidence_high=None,
        model_version="test",
    )
    agent = MagicMock()
    agent.run.return_value = fc
    return agent


def _build_test_graph(db: MarketDB, latest_lmp: float = 40.0):
    """Compile the graph with fetch + market-intel mocked, real risk/opt/resolve."""
    caiso_fetcher = MagicMock()
    weather_fetcher = MagicMock()
    market_intel = MagicMock()
    market_intel.run.return_value = MagicMock(raw_summary="")  # no EEA keywords
    forecaster = _mock_forecaster()
    optimizer = DispatchOptimizerAgent(battery=DEFAULT_BATTERY, node=DEFAULT_HUB_NODE)
    risk = RiskMonitorAgent()
    return build_graph(caiso_fetcher, weather_fetcher, forecaster, optimizer, risk, market_intel, db)


def _seed_state(latest_lmp: float = 40.0) -> SharedMarketState:
    # Pre-load recent_lmp_df so the fetch node skips live API calls.
    df = pd.DataFrame({"lmp": [latest_lmp]})
    return SharedMarketState(
        tick_time=datetime(2024, 7, 1),
        node=DEFAULT_HUB_NODE,
        latest_rt_lmp=latest_lmp,
        recent_lmp_df=df,
    )


def test_single_tick_completes_and_produces_dispatch() -> None:
    """A normal tick runs all nodes and yields an optimal dispatch schedule."""
    with MarketDB(":memory:") as db:
        graph = _build_test_graph(db)
        gs: GraphState = {"state": _seed_state(latest_lmp=40.0), "reoptimize_count": 0}
        out = graph.invoke(gs)

    state = out["state"]
    assert state.errors == []
    assert state.dispatch_result is not None
    assert "optimal" in state.dispatch_result.status.lower()
    # The forecast had an evening spread, so the optimizer should discharge.
    assert state.dispatch_result.discharge_mw.sum() > 0.0
    # finalize updated cumulative P&L and SoC carry-forward.
    assert state.cumulative_pnl_usd == state.dispatch_result.net_revenue_usd


def test_price_spike_tick_does_not_crash() -> None:
    """A spike LMP triggers RiskMonitor constraints/halt without breaking the graph."""
    with MarketDB(":memory:") as db:
        graph = _build_test_graph(db)
        gs: GraphState = {"state": _seed_state(latest_lmp=750.0), "reoptimize_count": 0}
        out = graph.invoke(gs)

    state = out["state"]
    # The graph completes; a risk event was detected for the spike.
    assert any(e.event_type == "PRICE_SPIKE" for e in state.risk_events)
