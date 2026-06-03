"""
SharedMarketState — single source of truth for all agents in a tick.

This dataclass is passed between agents within each LangGraph graph tick.
It is immutable within a single agent's execution (agents return updated copies),
enforcing the LangGraph pattern of returning new state rather than mutating in place.

The state is also checkpointed to DuckDB after each tick so the system can
resume from the latest consistent state after a crash or restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.agents.forecaster import PriceForecast
from src.agents.market_intel import MarketIntelSummary
from src.agents.risk_monitor import RiskEvent, RiskLevel
from src.optimization.battery_dispatch import DispatchConstraint, DispatchResult


@dataclass
class SharedMarketState:
    """Immutable snapshot of all agent inputs, outputs, and market data for one tick.

    Attributes:
        tick_time: Wall-clock time when this tick was initiated.
        node: CAISO hub node being traded.

        --- Data layer (populated by fetch step) ---
        latest_rt_lmp: Most recent real-time LMP in $/MWh (scalar).
        latest_da_lmp: Most recent day-ahead LMP for the current hour.
        recent_lmp_df: Last ~48 h of RT LMP data (for ramp detection).
        load_forecast_df: 7-day ahead load forecast.
        weather_forecast_df: Solar and meteorological forecast.

        --- Agent outputs ---
        price_forecast: Output from PriceForecasterAgent (may be None if not yet run).
        dispatch_result: Output from DispatchOptimizerAgent (may be None).
        risk_events: List of RiskEvents detected by RiskMonitorAgent this tick.
        market_intel: Structured market intelligence from MarketIntelAgent.

        --- Conflict resolution ---
        active_constraints: DispatchConstraints accumulated across agents this tick.
        is_halted: True if RiskMonitor has issued a HALT override.

        --- Carry-forward state ---
        current_soc_mwh: Current battery SoC in MWh (updated after each tick).
        cumulative_pnl_usd: Running P&L since simulation start.

        --- Metadata ---
        errors: List of error strings encountered during this tick.
        agent_logs: Dict mapping agent name → list of log message strings.
    """

    tick_time: datetime
    node: str

    # Market data
    latest_rt_lmp: float | None = None
    latest_da_lmp: float | None = None
    recent_lmp_df: pd.DataFrame | None = None
    load_forecast_df: pd.DataFrame | None = None
    weather_forecast_df: pd.DataFrame | None = None

    # Agent outputs
    price_forecast: PriceForecast | None = None
    dispatch_result: DispatchResult | None = None
    risk_events: list[RiskEvent] = field(default_factory=list)
    market_intel: MarketIntelSummary | None = None

    # Conflict resolution
    active_constraints: list[DispatchConstraint] = field(default_factory=list)
    is_halted: bool = False

    # Carry-forward
    current_soc_mwh: float = 2.0
    cumulative_pnl_usd: float = 0.0

    # Metadata
    errors: list[str] = field(default_factory=list)
    agent_logs: dict[str, list[str]] = field(default_factory=dict)

    def with_price_forecast(self, forecast: PriceForecast) -> "SharedMarketState":
        """Return a new state with the price forecast set."""
        return replace(self, price_forecast=forecast)

    def with_dispatch_result(self, result: DispatchResult) -> "SharedMarketState":
        """Return a new state with the dispatch result set."""
        return replace(self, dispatch_result=result)

    def with_risk_events(self, events: list[RiskEvent]) -> "SharedMarketState":
        """Return a new state with risk events appended and constraints merged."""
        new_constraints = self.active_constraints + [c for e in events for c in e.constraints]
        is_halted = any(e.level == RiskLevel.HALT for e in events)
        return replace(self, risk_events=events, active_constraints=new_constraints, is_halted=is_halted)

    def with_market_intel(self, intel: MarketIntelSummary) -> "SharedMarketState":
        """Return a new state with market intelligence set."""
        return replace(self, market_intel=intel)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state to a JSON-compatible dict for DuckDB checkpointing.

        Returns:
            Dict with all serializable fields (DataFrames as JSON strings).
        """
        def _arr(a: np.ndarray | None) -> list | None:
            return a.tolist() if a is not None else None

        def _df(df: pd.DataFrame | None) -> str | None:
            return df.to_json(orient="split") if df is not None else None

        def _con(c: DispatchConstraint) -> dict:
            return {
                "interval_idx": c.interval_idx, "variable": c.variable,
                "operator": c.operator, "value_mw": c.value_mw,
                "source": c.source, "reason": c.reason,
            }

        price_forecast_d = None
        if self.price_forecast is not None:
            f = self.price_forecast
            price_forecast_d = {
                "node": f.node,
                "times": [str(t) for t in f.times],
                "da_lmp_forecast": _arr(f.da_lmp_forecast),
                "rt_lmp_forecast": _arr(f.rt_lmp_forecast),
                "confidence_low": _arr(f.confidence_low),
                "confidence_high": _arr(f.confidence_high),
                "model_version": f.model_version,
            }

        dispatch_result_d = None
        if self.dispatch_result is not None:
            r = self.dispatch_result
            dispatch_result_d = {
                "charge_mw": _arr(r.charge_mw),
                "discharge_mw": _arr(r.discharge_mw),
                "soc_mwh": _arr(r.soc_mwh),
                "net_revenue_usd": r.net_revenue_usd,
                "solve_time_s": r.solve_time_s,
                "status": r.status,
                "active_constraints": [_con(c) for c in r.active_constraints],
            }

        risk_events_d = [
            {
                "event_type": e.event_type,
                "level": e.level.value,
                "description": e.description,
                "recommended_action": e.recommended_action,
                "raw_value": e.raw_value,
                "constraints": [_con(c) for c in e.constraints],
            }
            for e in self.risk_events
        ]

        market_intel_d = None
        if self.market_intel is not None:
            m = self.market_intel
            market_intel_d = {
                "eea_level": m.eea_level,
                "has_curtailment_event": m.has_curtailment_event,
                "curtailment_mw_estimated": m.curtailment_mw_estimated,
                "has_major_outage": m.has_major_outage,
                "outage_summary": m.outage_summary,
                "weather_risk": m.weather_risk,
                "dispatch_recommendation": m.dispatch_recommendation,
                "key_events": m.key_events,
                "raw_summary": m.raw_summary,
                "sources": m.sources,
            }

        return {
            "tick_time": self.tick_time.isoformat(),
            "node": self.node,
            "latest_rt_lmp": self.latest_rt_lmp,
            "latest_da_lmp": self.latest_da_lmp,
            "recent_lmp_df": _df(self.recent_lmp_df),
            "load_forecast_df": _df(self.load_forecast_df),
            "weather_forecast_df": _df(self.weather_forecast_df),
            "price_forecast": price_forecast_d,
            "dispatch_result": dispatch_result_d,
            "risk_events": risk_events_d,
            "market_intel": market_intel_d,
            "active_constraints": [_con(c) for c in self.active_constraints],
            "is_halted": self.is_halted,
            "current_soc_mwh": self.current_soc_mwh,
            "cumulative_pnl_usd": self.cumulative_pnl_usd,
            "errors": list(self.errors),
            "agent_logs": {k: list(v) for k, v in self.agent_logs.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SharedMarketState":
        """Deserialize a state from a DuckDB-checkpointed dict.

        Args:
            data: Dict produced by to_dict().

        Returns:
            SharedMarketState instance.
        """
        def _con(d: dict) -> DispatchConstraint:
            return DispatchConstraint(
                d["interval_idx"], d["variable"], d["operator"],
                d["value_mw"], d["source"], d.get("reason", ""),
            )

        def _df(j: str | None) -> pd.DataFrame | None:
            return pd.read_json(j, orient="split") if j is not None else None

        price_forecast = None
        if data.get("price_forecast") is not None:
            fd = data["price_forecast"]
            price_forecast = PriceForecast(
                node=fd["node"],
                times=pd.DatetimeIndex(pd.to_datetime(fd["times"])),
                da_lmp_forecast=np.array(fd["da_lmp_forecast"]) if fd.get("da_lmp_forecast") is not None else np.array([]),
                rt_lmp_forecast=np.array(fd["rt_lmp_forecast"]) if fd.get("rt_lmp_forecast") is not None else None,
                confidence_low=np.array(fd["confidence_low"]) if fd.get("confidence_low") is not None else None,
                confidence_high=np.array(fd["confidence_high"]) if fd.get("confidence_high") is not None else None,
                model_version=fd["model_version"],
            )

        dispatch_result = None
        if data.get("dispatch_result") is not None:
            rd = data["dispatch_result"]
            dispatch_result = DispatchResult(
                charge_mw=np.array(rd["charge_mw"]),
                discharge_mw=np.array(rd["discharge_mw"]),
                soc_mwh=np.array(rd["soc_mwh"]),
                net_revenue_usd=rd["net_revenue_usd"],
                solve_time_s=rd["solve_time_s"],
                status=rd["status"],
                active_constraints=[_con(c) for c in rd.get("active_constraints", [])],
            )

        risk_events = [
            RiskEvent(
                event_type=e["event_type"],
                level=RiskLevel(e["level"]),
                description=e["description"],
                recommended_action=e["recommended_action"],
                raw_value=e.get("raw_value"),
                constraints=[_con(c) for c in e.get("constraints", [])],
            )
            for e in data.get("risk_events", [])
        ]

        market_intel = None
        if data.get("market_intel") is not None:
            md = data["market_intel"]
            market_intel = MarketIntelSummary(
                eea_level=md.get("eea_level"),
                has_curtailment_event=md.get("has_curtailment_event", False),
                curtailment_mw_estimated=md.get("curtailment_mw_estimated"),
                has_major_outage=md.get("has_major_outage", False),
                outage_summary=md.get("outage_summary"),
                weather_risk=md.get("weather_risk", "none"),
                dispatch_recommendation=md.get("dispatch_recommendation", ""),
                key_events=md.get("key_events", []),
                raw_summary=md.get("raw_summary", ""),
                sources=md.get("sources", []),
            )

        return cls(
            tick_time=datetime.fromisoformat(data["tick_time"]),
            node=data["node"],
            latest_rt_lmp=data.get("latest_rt_lmp"),
            latest_da_lmp=data.get("latest_da_lmp"),
            recent_lmp_df=_df(data.get("recent_lmp_df")),
            load_forecast_df=_df(data.get("load_forecast_df")),
            weather_forecast_df=_df(data.get("weather_forecast_df")),
            price_forecast=price_forecast,
            dispatch_result=dispatch_result,
            risk_events=risk_events,
            market_intel=market_intel,
            active_constraints=[_con(c) for c in data.get("active_constraints", [])],
            is_halted=data.get("is_halted", False),
            current_soc_mwh=data.get("current_soc_mwh", 2.0),
            cumulative_pnl_usd=data.get("cumulative_pnl_usd", 0.0),
            errors=data.get("errors", []),
            agent_logs=data.get("agent_logs", {}),
        )
