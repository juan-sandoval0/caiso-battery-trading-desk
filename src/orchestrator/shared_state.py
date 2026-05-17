"""
SharedMarketState — single source of truth for all agents in a tick.

This dataclass is passed between agents within each LangGraph graph tick.
It is immutable within a single agent's execution (agents return updated copies),
enforcing the LangGraph pattern of returning new state rather than mutating in place.

The state is also checkpointed to DuckDB after each tick so the system can
resume from the latest consistent state after a crash or restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.agents.forecaster import PriceForecast
from src.agents.market_intel import MarketIntelSummary
from src.agents.risk_monitor import RiskEvent
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
        raise NotImplementedError(
            # Use dataclasses.replace(self, price_forecast=forecast)
        )

    def with_dispatch_result(self, result: DispatchResult) -> "SharedMarketState":
        """Return a new state with the dispatch result set."""
        raise NotImplementedError()

    def with_risk_events(self, events: list[RiskEvent]) -> "SharedMarketState":
        """Return a new state with risk events appended and constraints merged."""
        raise NotImplementedError(
            # Merge new constraints from events into active_constraints.
            # Set is_halted = True if any event has level == RiskLevel.HALT.
        )

    def with_market_intel(self, intel: MarketIntelSummary) -> "SharedMarketState":
        """Return a new state with market intelligence set."""
        raise NotImplementedError()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state to a JSON-compatible dict for DuckDB checkpointing.

        Returns:
            Dict with all serializable fields (DataFrames as JSON strings).
        """
        raise NotImplementedError(
            # Convert DataFrames to .to_json(). Convert numpy arrays to lists.
            # Convert dataclass fields to dicts recursively.
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SharedMarketState":
        """Deserialize a state from a DuckDB-checkpointed dict.

        Args:
            data: Dict produced by to_dict().

        Returns:
            SharedMarketState instance.
        """
        raise NotImplementedError(
            # Reverse of to_dict(): reconstruct DataFrames from JSON strings, etc.
        )
