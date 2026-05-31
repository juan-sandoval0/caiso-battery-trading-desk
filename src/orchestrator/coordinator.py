"""
Coordinator — LangGraph supervisor graph for the multi-agent trading desk.

Defines the LangGraph StateGraph that orchestrates all four agents:
    1. fetch_data       — pull latest CAISO + weather data
    2. run_market_intel — MarketIntelAgent (qualitative context)
    3. forecast_prices  — PriceForecasterAgent
    4. monitor_risk     — RiskMonitorAgent (pre-optimization check)
    5. optimize_dispatch — DispatchOptimizerAgent
    6. resolve_conflicts — ConflictResolver (may re-invoke optimizer)
    7. finalize         — log, checkpoint state, notify dashboard

The graph ticks every 5 minutes in live mode (synchronized with CAISO RT intervals).
In backtest mode, the graph is invoked once per simulated interval without sleeping.

LangGraph graph topology:
    START → fetch_data → run_market_intel → forecast_prices
          → monitor_risk → optimize_dispatch → resolve_conflicts → finalize → END

Conditional edge from resolve_conflicts:
    If constraints were added AFTER optimize_dispatch ran → re_optimize → finalize
    Otherwise → finalize directly
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any, TypedDict

import structlog

from src.orchestrator.shared_state import SharedMarketState

log = structlog.get_logger()


class GraphState(TypedDict):
    """LangGraph state wrapper: holds the SharedMarketState plus loop bookkeeping."""
    state: SharedMarketState
    reoptimize_count: int


def build_graph(
    caiso_fetcher: Any,
    weather_fetcher: Any,
    forecaster_agent: Any,
    optimizer_agent: Any,
    risk_monitor_agent: Any,
    market_intel_agent: Any,
    db: Any,
) -> Any:
    """Construct and compile the LangGraph StateGraph.

    Args:
        caiso_fetcher: CAISOFetcher instance.
        weather_fetcher: WeatherFetcher instance.
        forecaster_agent: PriceForecasterAgent instance.
        optimizer_agent: DispatchOptimizerAgent instance.
        risk_monitor_agent: RiskMonitorAgent instance.
        market_intel_agent: MarketIntelAgent instance.
        db: MarketDB instance for state checkpointing.

    Returns:
        A compiled LangGraph CompiledGraph ready for .ainvoke() or .invoke().
    """
    from langgraph.graph import StateGraph, END

    graph = StateGraph(GraphState)

    graph.add_node("fetch_data", _make_fetch_node(caiso_fetcher, weather_fetcher, db))
    graph.add_node("run_market_intel", _make_market_intel_node(market_intel_agent))
    graph.add_node("forecast_prices", _make_forecast_node(forecaster_agent, db, weather_fetcher, caiso_fetcher))
    graph.add_node("monitor_risk", _make_risk_node(risk_monitor_agent, db))
    graph.add_node("optimize_dispatch", _make_optimize_node(optimizer_agent, db))
    graph.add_node("resolve_conflicts", _make_resolve_node())
    graph.add_node("finalize", _make_finalize_node(db))

    graph.set_entry_point("fetch_data")
    graph.add_edge("fetch_data", "run_market_intel")
    graph.add_edge("run_market_intel", "forecast_prices")
    graph.add_edge("forecast_prices", "monitor_risk")
    graph.add_edge("monitor_risk", "optimize_dispatch")
    graph.add_edge("optimize_dispatch", "resolve_conflicts")
    graph.add_conditional_edges(
        "resolve_conflicts",
        _needs_reoptimize,
        {"reoptimize": "optimize_dispatch", "done": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()


class TradingDeskCoordinator:
    """High-level coordinator that manages the LangGraph tick loop.

    Handles:
    - Building the compiled graph from agent instances
    - Running the graph tick-by-tick in live mode (5-min cadence)
    - Running the graph over historical intervals in backtest mode
    - State checkpointing and recovery

    Usage::

        coordinator = TradingDeskCoordinator.from_config(settings)
        # Live paper-trading:
        asyncio.run(coordinator.run_live(node="TH_NP15_GEN-APND"))
        # Backtest:
        results = coordinator.run_backtest(start="2024-01-01", end="2024-12-31")
    """

    def __init__(
        self,
        graph: Any,
        db: Any,
        initial_state: SharedMarketState,
    ) -> None:
        """Initialize with a compiled graph and initial state.

        Args:
            graph: Compiled LangGraph graph from build_graph().
            db: MarketDB instance.
            initial_state: Starting SharedMarketState (SoC, P&L, etc.).
        """
        self._graph = graph
        self._db = db
        self._state = initial_state

    @classmethod
    def from_config(cls, settings: Any) -> "TradingDeskCoordinator":
        """Factory method: instantiate all agents and build graph from settings.

        Args:
            settings: Settings instance from src.config.settings.

        Returns:
            Fully initialized TradingDeskCoordinator.
        """
        from pathlib import Path

        from src.agents.forecaster import PriceForecasterAgent
        from src.agents.market_intel import MarketIntelAgent
        from src.agents.optimizer import DispatchOptimizerAgent
        from src.agents.risk_monitor import RiskMonitorAgent
        from src.config.battery import DEFAULT_BATTERY
        from src.config.nodes import node_short_name
        from src.data.caiso_fetcher import CAISOFetcher
        from src.data.db import MarketDB
        from src.data.weather_fetcher import WeatherFetcher

        db = MarketDB(str(settings.duckdb_path))
        caiso_fetcher = CAISOFetcher(api_key=settings.gridstatus_api_key)
        weather_fetcher = WeatherFetcher()

        forecaster = PriceForecasterAgent(node=settings.default_node)
        short = node_short_name(settings.default_node)
        da_path = Path("models/artifacts") / f"da_{short}.joblib"
        rt_path = Path("models/artifacts") / f"rt_{short}.joblib"
        if da_path.exists():
            forecaster.load_models(da_path=str(da_path))
        if rt_path.exists():
            forecaster.load_models(rt_path=str(rt_path))

        optimizer = DispatchOptimizerAgent(battery=DEFAULT_BATTERY, node=settings.default_node)
        risk_monitor = RiskMonitorAgent()
        market_intel = MarketIntelAgent(
            api_key=settings.openrouter_api_key,
            caiso_fetcher=caiso_fetcher,
            base_url=settings.openrouter_base_url,
        )

        graph = build_graph(
            caiso_fetcher, weather_fetcher, forecaster,
            optimizer, risk_monitor, market_intel, db,
        )
        initial_state = SharedMarketState(tick_time=datetime.utcnow(), node=settings.default_node)
        return cls(graph, db, initial_state)

    async def run_live(self, node: str, interval_minutes: int = 5) -> None:
        """Run the agent graph indefinitely, ticking every interval_minutes.

        Args:
            node: CAISO hub node to trade.
            interval_minutes: Tick cadence (5 matches CAISO RT dispatch intervals).
        """
        log.info("paper_trade_start", node=node, interval_minutes=interval_minutes)
        while True:
            self._state = replace(self._state, tick_time=datetime.utcnow(), node=node)
            gs: GraphState = {"state": self._state, "reoptimize_count": 0}
            try:
                result = self._graph.invoke(gs)
                self._state = result["state"]
                log.info(
                    "tick_complete",
                    node=node,
                    pnl=self._state.cumulative_pnl_usd,
                    is_halted=self._state.is_halted,
                    errors=len(self._state.errors),
                )
            except Exception as exc:
                log.error("tick_failed", error=str(exc))

            await asyncio.sleep(interval_minutes * 60)

    def run_backtest(
        self,
        start: str,
        end: str,
        node: str,
    ) -> list[dict[str, Any]]:
        """Run backtest over historical intervals and return per-day P&L records.

        Loads historical LMP data from DuckDB for each day, pre-populates the
        SharedMarketState so the fetch_data node skips live API calls, then
        runs the full multi-agent graph.

        Args:
            start: ISO date string (inclusive).
            end: ISO date string (inclusive).
            node: CAISO hub node.

        Returns:
            List of daily P&L dicts with keys: date, revenue_usd, vs_naive_usd,
            vs_perfect_usd, cycles, final_soc_mwh.
        """
        from datetime import date, timedelta

        results: list[dict[str, Any]] = []
        cur = date.fromisoformat(start)
        end_d = date.fromisoformat(end)

        while cur <= end_d:
            date_str = str(cur)
            log.info("backtest_day", date=date_str, node=node)
            try:
                # Load historical LMP data for this day from DuckDB
                lmp_df = self._db.query(
                    f"SELECT time, lmp FROM lmp WHERE node = '{node}' "
                    f"AND market = 'REAL_TIME_5_MIN' "
                    f"AND time::DATE = '{date_str}' ORDER BY time"
                )

                if lmp_df.empty:
                    cur += timedelta(days=1)
                    continue

                latest_lmp = float(lmp_df["lmp"].iloc[-1])
                tick_state = replace(
                    self._state,
                    tick_time=datetime.combine(cur, datetime.min.time()),
                    node=node,
                    latest_rt_lmp=latest_lmp,
                    recent_lmp_df=lmp_df,  # pre-loaded → fetch_data will skip API call
                )

                gs: GraphState = {"state": tick_state, "reoptimize_count": 0}
                result_gs = self._graph.invoke(gs)
                tick_state = result_gs["state"]

                rev = 0.0
                cycles = 0.0
                final_soc = tick_state.current_soc_mwh
                if tick_state.dispatch_result is not None:
                    dr = tick_state.dispatch_result
                    rev = dr.net_revenue_usd
                    cycles = float(dr.discharge_mw.sum()) / max(self._state.current_soc_mwh, 1.0)
                    final_soc = float(dr.soc_mwh[-1]) if len(dr.soc_mwh) > 0 else final_soc

                results.append({
                    "date": date_str,
                    "revenue_usd": rev,
                    "cycles": cycles,
                    "final_soc_mwh": final_soc,
                    "is_halted": tick_state.is_halted,
                    "errors": len(tick_state.errors),
                })

                # Carry SoC forward to next day
                self._state = replace(self._state, current_soc_mwh=final_soc)

            except Exception as exc:
                log.error("backtest_day_failed", date=date_str, error=str(exc))
                results.append({"date": date_str, "revenue_usd": 0.0, "error": str(exc)})

            cur += timedelta(days=1)

        return results


# ---------------------------------------------------------------------------
# Private node factory functions (return callables consumed by LangGraph)
# ---------------------------------------------------------------------------

def _make_fetch_node(caiso_fetcher: Any, weather_fetcher: Any, db: Any) -> Any:
    """Return a LangGraph node function that fetches latest market data."""

    def fetch_node(gs: GraphState) -> dict:
        s = gs["state"]

        # Backtest mode: data is pre-loaded, skip live API calls
        if s.recent_lmp_df is not None:
            return {"state": s}

        new_s = s
        try:
            lmp_df = caiso_fetcher.get_lmp_latest(
                market="REAL_TIME_5_MIN",
                nodes=[s.node],
            )
            if not lmp_df.empty:
                node_df = lmp_df[lmp_df["node"] == s.node] if "node" in lmp_df.columns else lmp_df
                if not node_df.empty:
                    latest_lmp = float(node_df["lmp"].iloc[-1])
                    new_s = replace(new_s, latest_rt_lmp=latest_lmp, recent_lmp_df=node_df)
        except Exception as exc:
            new_s = replace(new_s, errors=new_s.errors + [f"fetch_rt_lmp: {exc}"])

        try:
            da_df = caiso_fetcher.get_lmp_latest(
                market="DAY_AHEAD_HOURLY",
                nodes=[s.node],
            )
            if not da_df.empty:
                node_da = da_df[da_df["node"] == s.node] if "node" in da_df.columns else da_df
                if not node_da.empty:
                    latest_da = float(node_da["lmp"].iloc[-1])
                    new_s = replace(new_s, latest_da_lmp=latest_da)
        except Exception as exc:
            new_s = replace(new_s, errors=new_s.errors + [f"fetch_da_lmp: {exc}"])

        return {"state": new_s}

    return fetch_node


def _make_market_intel_node(market_intel_agent: Any) -> Any:
    """Return a LangGraph node function that runs the MarketIntelAgent."""

    def market_intel_node(gs: GraphState) -> dict:
        s = gs["state"]
        try:
            intel = market_intel_agent.run()
            new_s = s.with_market_intel(intel)
        except Exception as exc:
            new_s = replace(s, errors=s.errors + [f"market_intel: {exc}"])
        return {"state": new_s}

    return market_intel_node


def _make_forecast_node(
    forecaster_agent: Any, db: Any, weather_fetcher: Any, caiso_fetcher: Any
) -> Any:
    """Return a LangGraph node function that runs the PriceForecasterAgent."""

    def forecast_node(gs: GraphState) -> dict:
        s = gs["state"]
        try:
            forecast = forecaster_agent.run(
                db=db, weather_fetcher=weather_fetcher, caiso_fetcher=caiso_fetcher
            )
            new_s = s.with_price_forecast(forecast)
        except Exception as exc:
            new_s = replace(s, errors=s.errors + [f"forecast: {exc}"])
        return {"state": new_s}

    return forecast_node


def _make_risk_node(risk_monitor_agent: Any, db: Any) -> Any:
    """Return a LangGraph node function that runs the RiskMonitorAgent."""

    def risk_node(gs: GraphState) -> dict:
        s = gs["state"]
        events = risk_monitor_agent.run(state=s, db=db)
        new_s = s.with_risk_events(events)
        return {"state": new_s}

    return risk_node


def _make_optimize_node(optimizer_agent: Any, db: Any) -> Any:
    """Return a LangGraph node function that runs the DispatchOptimizerAgent."""

    def optimize_node(gs: GraphState) -> dict:
        s = gs["state"]
        if s.is_halted or s.price_forecast is None:
            return {"state": s}
        try:
            result = optimizer_agent.run(state=s, db=db)
            new_s = s.with_dispatch_result(result)
        except Exception as exc:
            new_s = replace(s, errors=s.errors + [f"optimize: {exc}"])
        return {"state": new_s}

    return optimize_node


def _make_resolve_node() -> Any:
    """Return a LangGraph node function that runs ConflictResolver."""
    from src.orchestrator.conflict_resolver import resolve_conflicts

    def resolve_node(gs: GraphState) -> dict:
        s = gs["state"]
        count = gs.get("reoptimize_count", 0) + 1

        if s.dispatch_result is None:
            return {"state": s, "reoptimize_count": count}

        resolved = resolve_conflicts(s.dispatch_result, s.risk_events, s.is_halted)
        new_s = replace(s, dispatch_result=resolved)
        return {"state": new_s, "reoptimize_count": count}

    return resolve_node


def _make_finalize_node(db: Any) -> Any:
    """Return a LangGraph node function that checkpoints state and logs P&L."""

    def finalize_node(gs: GraphState) -> dict:
        s = gs["state"]

        # Update cumulative P&L from this tick
        tick_revenue = 0.0
        if s.dispatch_result is not None:
            tick_revenue = s.dispatch_result.net_revenue_usd

        new_soc = s.current_soc_mwh
        if s.dispatch_result is not None and len(s.dispatch_result.soc_mwh) > 0:
            new_soc = float(s.dispatch_result.soc_mwh[-1])

        new_s = replace(
            s,
            cumulative_pnl_usd=s.cumulative_pnl_usd + tick_revenue,
            current_soc_mwh=new_soc,
        )

        # Checkpoint to DuckDB
        try:
            db.log_agent_decision(
                agent="Coordinator",
                action="TICK_COMPLETE",
                rationale=f"tick pnl=${tick_revenue:.2f}, cumulative=${new_s.cumulative_pnl_usd:.2f}",
                metadata={
                    "is_halted": new_s.is_halted,
                    "n_risk_events": len(new_s.risk_events),
                    "soc_mwh": new_soc,
                    "errors": new_s.errors,
                },
            )
        except Exception:
            pass

        return {"state": new_s}

    return finalize_node


def _needs_reoptimize(gs: GraphState) -> str:
    """Conditional edge function: returns 'reoptimize' or 'done'.

    Re-optimization is needed if risk events added constraints AFTER the
    initial optimizer run. This prevents stale schedules from being executed.

    Args:
        gs: Current GraphState after resolve_conflicts ran.

    Returns:
        'reoptimize' if a second LP solve is needed, 'done' otherwise.
    """
    s = gs["state"]
    count = gs.get("reoptimize_count", 0)

    # Prevent infinite re-optimization loops
    if count >= 2:
        return "done"

    if s.is_halted:
        return "done"

    # Re-optimize if any non-optimizer constraints are present
    non_optimizer = [c for c in s.active_constraints if c.source != "optimizer"]
    if non_optimizer:
        return "reoptimize"

    return "done"
