"""
RiskMonitorAgent — anomaly detection and emergency override agent.

Continuously monitors market conditions and system state for events that
require overriding the optimizer's dispatch schedule. This agent has the
HIGHEST priority in the conflict resolution hierarchy.

Monitored conditions:
    Price spikes   — LMP > $500/MWh (CAISO hard cap is $2,000/MWh)
    Negative prices — LMP < -$50/MWh (curtailment signal; opportunistic charging)
    Grid emergency  — CAISO Energy Emergency Alert (EEA) level 1/2/3
    Ramp events     — LMP changes > $200/MWh within a single 5-min interval
    SOC violation   — Forecast schedule would breach SoC bounds
    Solver failure  — Optimizer returned infeasible or error status

On detection, the agent either:
    HALT     — prevents all charge/discharge (e.g., during EEA3)
    OVERRIDE — replaces optimizer schedule with a conservative safe mode
    CONSTRAIN — injects a DispatchConstraint into the optimizer's next solve
    ALERT    — logs the event but takes no dispatch action
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.optimization.battery_dispatch import DispatchConstraint


class RiskLevel(Enum):
    """Severity of a detected risk event."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    HALT = "halt"


@dataclass
class RiskEvent:
    """A single detected risk event.

    Attributes:
        event_type: Short string tag (e.g., 'PRICE_SPIKE', 'EEA3').
        level: Severity level.
        description: Human-readable description of the event.
        recommended_action: What the agent recommends ('HALT', 'CONSTRAIN', 'ALERT').
        constraints: DispatchConstraints to inject if action is 'CONSTRAIN'.
        raw_value: The triggering value (e.g., the spike LMP).
    """

    event_type: str
    level: RiskLevel
    description: str
    recommended_action: str
    constraints: list[DispatchConstraint] = field(default_factory=list)
    raw_value: float | None = None


# Threshold constants — tune based on NP15 historical volatility
_SPIKE_THRESHOLD_MWH: float = 500.0
_NEGATIVE_THRESHOLD_MWH: float = -50.0
_RAMP_THRESHOLD_MWH: float = 200.0
_EEA_KEYWORDS: list[str] = ["EEA1", "EEA2", "EEA3", "Energy Emergency Alert"]


class RiskMonitorAgent:
    """Detects market anomalies and injects override constraints into the dispatch optimizer.

    This agent runs synchronously before the optimizer in each graph tick so
    it can block dangerous schedules before they are committed.

    Usage::

        monitor = RiskMonitorAgent(price_spike_threshold=500.0)
        events = monitor.run(state=shared_state, db=market_db)
        # Events are also written to state.risk_events
    """

    def __init__(
        self,
        price_spike_threshold: float = _SPIKE_THRESHOLD_MWH,
        negative_price_threshold: float = _NEGATIVE_THRESHOLD_MWH,
        ramp_threshold: float = _RAMP_THRESHOLD_MWH,
    ) -> None:
        """Initialize the monitor with configurable thresholds.

        Args:
            price_spike_threshold: LMP above this triggers a spike event ($/MWh).
            negative_price_threshold: LMP below this triggers a negative price event.
            ramp_threshold: Absolute intra-interval LMP change triggering ramp event.
        """
        raise NotImplementedError(
            # Store thresholds as self._spike_thr, self._neg_thr, self._ramp_thr.
            # Initialize self._event_history: list[RiskEvent] = [].
        )

    def run(self, state: Any, db: Any) -> list[RiskEvent]:
        """Execute one monitoring tick; return detected risk events.

        Detected events are written to state.risk_events and logged to db.
        Any HALT or CONSTRAIN events also populate state.active_constraints.

        Args:
            state: SharedMarketState with latest LMP snapshot and market intel.
            db: MarketDB for logging detected events.

        Returns:
            List of RiskEvent objects detected in this tick (may be empty).
        """
        raise NotImplementedError(
            # 1. Detect price spike: check state.latest_rt_lmp against _spike_thr.
            # 2. Detect negative prices: check against _neg_thr.
            # 3. Detect ramp events: diff consecutive LMP values in state.recent_lmp.
            # 4. Check state.market_intel_summary for EEA keywords.
            # 5. Validate state.dispatch_result SoC trajectory for bound violations.
            # 6. Aggregate events, update state, log to db, return events list.
        )

    def _detect_price_spike(self, lmp: float, interval_idx: int) -> RiskEvent | None:
        """Return a RiskEvent if lmp exceeds the spike threshold.

        Args:
            lmp: Current LMP value in $/MWh.
            interval_idx: Current time step index.

        Returns:
            RiskEvent with CONSTRAIN action (halt discharge), or None.
        """
        raise NotImplementedError(
            # if lmp > self._spike_thr: return RiskEvent(..., recommended_action='CONSTRAIN',
            #   constraints=[DispatchConstraint(interval_idx, 'discharge', 'eq', 0, 'RiskMonitor')])
        )

    def _detect_negative_prices(self, lmp: float, interval_idx: int) -> RiskEvent | None:
        """Return a RiskEvent if lmp is below the negative threshold.

        Negative prices signal curtailment — the battery should charge maximally.

        Args:
            lmp: Current LMP value in $/MWh.
            interval_idx: Current time step index.

        Returns:
            RiskEvent with CONSTRAIN action (force maximum charge), or None.
        """
        raise NotImplementedError(
            # if lmp < self._neg_thr: return RiskEvent(..., recommended_action='CONSTRAIN',
            #   constraints=[DispatchConstraint(interval_idx, 'charge', 'eq',
            #     battery.max_charge_mw, 'RiskMonitor')])
        )

    def _detect_eea(self, market_intel_text: str) -> RiskEvent | None:
        """Scan market intelligence text for CAISO Energy Emergency Alert keywords.

        Args:
            market_intel_text: Raw text from MarketIntelAgent summary.

        Returns:
            RiskEvent with HALT action if EEA detected, or None.
        """
        raise NotImplementedError(
            # Check for any string in _EEA_KEYWORDS appearing in market_intel_text.
            # EEA3 → HALT; EEA1/EEA2 → WARNING + CONSTRAIN (reduce discharge).
        )

    def _validate_soc_trajectory(
        self, soc_mwh: np.ndarray, soc_min: float, soc_max: float
    ) -> RiskEvent | None:
        """Detect if the proposed SoC trajectory violates physical bounds.

        Args:
            soc_mwh: Proposed SoC array from the optimizer.
            soc_min: Minimum SoC in MWh.
            soc_max: Maximum SoC in MWh.

        Returns:
            RiskEvent with OVERRIDE action if violation detected, or None.
        """
        raise NotImplementedError(
            # Check if any soc_mwh[i] < soc_min or > soc_max.
            # This should not happen if solver is correct, but acts as a safety net.
        )

    def get_langchain_tools(self) -> list[Any]:
        """Return LangChain Tool definitions for the orchestrator.

        Returns:
            List of Tool objects: 'check_risk_status', 'get_risk_event_history'.
        """
        raise NotImplementedError()
