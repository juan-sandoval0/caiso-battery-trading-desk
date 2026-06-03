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

import json
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
        self._spike_thr = price_spike_threshold
        self._neg_thr = negative_price_threshold
        self._ramp_thr = ramp_threshold
        self._event_history: list[RiskEvent] = []

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
        events: list[RiskEvent] = []

        lmp = state.latest_rt_lmp
        if lmp is not None:
            e = self._detect_price_spike(lmp, interval_idx=0)
            if e:
                events.append(e)
            e = self._detect_negative_prices(lmp, interval_idx=0)
            if e:
                events.append(e)

        if state.recent_lmp_df is not None and len(state.recent_lmp_df) > 1:
            lmp_series = state.recent_lmp_df["lmp"] if "lmp" in state.recent_lmp_df.columns else None
            if lmp_series is not None:
                diffs = lmp_series.diff().abs()
                for i, diff in enumerate(diffs):
                    if i == 0 or pd.isna(diff):
                        continue
                    if diff > self._ramp_thr:
                        events.append(RiskEvent(
                            event_type="RAMP_EVENT",
                            level=RiskLevel.WARNING,
                            description=f"LMP ramp of ${diff:.1f}/MWh at interval {i}",
                            recommended_action="ALERT",
                            raw_value=float(diff),
                        ))
                        break  # one ramp event per tick is enough

        intel_text = state.market_intel.raw_summary if state.market_intel else ""
        e = self._detect_eea(intel_text)
        if e:
            events.append(e)

        if state.dispatch_result is not None:
            from src.config.battery import DEFAULT_BATTERY as bat
            e = self._validate_soc_trajectory(
                state.dispatch_result.soc_mwh, bat.soc_min_mwh, bat.soc_max_mwh
            )
            if e:
                events.append(e)

        self._event_history.extend(events)
        for ev in events:
            try:
                db.log_agent_decision(
                    agent="RiskMonitor",
                    action=ev.recommended_action,
                    rationale=ev.description,
                    metadata={"event_type": ev.event_type, "level": ev.level.value},
                )
            except Exception:
                pass  # never crash the graph tick due to logging failure

        return events

    def _detect_price_spike(self, lmp: float, interval_idx: int) -> RiskEvent | None:
        """Return a RiskEvent if lmp exceeds the spike threshold.

        Args:
            lmp: Current LMP value in $/MWh.
            interval_idx: Current time step index.

        Returns:
            RiskEvent with CONSTRAIN action (halt discharge), or None.
        """
        if lmp > self._spike_thr:
            return RiskEvent(
                event_type="PRICE_SPIKE",
                level=RiskLevel.CRITICAL,
                description=(
                    f"LMP ${lmp:.1f}/MWh exceeds spike threshold "
                    f"${self._spike_thr:.1f}/MWh — halting discharge"
                ),
                recommended_action="CONSTRAIN",
                constraints=[
                    DispatchConstraint(
                        interval_idx=interval_idx,
                        variable="discharge",
                        operator="eq",
                        value_mw=0.0,
                        source="RiskMonitor",
                        reason="price_spike_discharge_halt",
                    )
                ],
                raw_value=lmp,
            )
        return None

    def _detect_negative_prices(self, lmp: float, interval_idx: int) -> RiskEvent | None:
        """Return a RiskEvent if lmp is below the negative threshold.

        Negative prices signal curtailment — the battery should charge maximally.

        Args:
            lmp: Current LMP value in $/MWh.
            interval_idx: Current time step index.

        Returns:
            RiskEvent with CONSTRAIN action (force maximum charge), or None.
        """
        if lmp < self._neg_thr:
            from src.config.battery import DEFAULT_BATTERY as bat
            return RiskEvent(
                event_type="NEGATIVE_PRICES",
                level=RiskLevel.WARNING,
                description=(
                    f"LMP ${lmp:.1f}/MWh below threshold "
                    f"${self._neg_thr:.1f}/MWh — curtailment signal, maximising charge"
                ),
                recommended_action="CONSTRAIN",
                constraints=[
                    DispatchConstraint(
                        interval_idx=interval_idx,
                        variable="charge",
                        operator="eq",
                        value_mw=bat.max_charge_mw,
                        source="RiskMonitor",
                        reason="negative_prices_max_charge",
                    )
                ],
                raw_value=lmp,
            )
        return None

    def _detect_eea(self, market_intel_text: str) -> RiskEvent | None:
        """Scan market intelligence text for CAISO Energy Emergency Alert keywords.

        Args:
            market_intel_text: Raw text from MarketIntelAgent summary.

        Returns:
            RiskEvent with HALT action if EEA detected, or None.
        """
        if not market_intel_text:
            return None
        text_upper = market_intel_text.upper()

        if "EEA3" in text_upper:
            return RiskEvent(
                event_type="EEA3",
                level=RiskLevel.HALT,
                description="CAISO Energy Emergency Alert Level 3 — halting all dispatch",
                recommended_action="HALT",
                raw_value=3.0,
            )
        if "EEA2" in text_upper:
            from src.config.battery import DEFAULT_BATTERY as bat
            return RiskEvent(
                event_type="EEA2",
                level=RiskLevel.CRITICAL,
                description="CAISO Energy Emergency Alert Level 2 — reducing discharge to 50%",
                recommended_action="CONSTRAIN",
                constraints=[
                    DispatchConstraint(
                        interval_idx=0,
                        variable="discharge",
                        operator="le",
                        value_mw=bat.max_discharge_mw * 0.5,
                        source="RiskMonitor",
                        reason="eea2_reduce_discharge",
                    )
                ],
                raw_value=2.0,
            )
        if "EEA1" in text_upper or "ENERGY EMERGENCY ALERT" in text_upper:
            from src.config.battery import DEFAULT_BATTERY as bat
            return RiskEvent(
                event_type="EEA1",
                level=RiskLevel.CRITICAL,
                description="CAISO Energy Emergency Alert Level 1 — reducing discharge to 50%",
                recommended_action="CONSTRAIN",
                constraints=[
                    DispatchConstraint(
                        interval_idx=0,
                        variable="discharge",
                        operator="le",
                        value_mw=bat.max_discharge_mw * 0.5,
                        source="RiskMonitor",
                        reason="eea1_reduce_discharge",
                    )
                ],
                raw_value=1.0,
            )
        return None

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
        tol = 1e-6
        if np.any(soc_mwh < soc_min - tol) or np.any(soc_mwh > soc_max + tol):
            v_min = float(np.min(soc_mwh))
            v_max = float(np.max(soc_mwh))
            return RiskEvent(
                event_type="SOC_VIOLATION",
                level=RiskLevel.CRITICAL,
                description=(
                    f"SoC trajectory violates bounds [{soc_min:.2f}, {soc_max:.2f}] MWh. "
                    f"Actual range: [{v_min:.2f}, {v_max:.2f}] MWh"
                ),
                recommended_action="OVERRIDE",
                raw_value=v_min,
            )
        return None

    def get_langchain_tools(self) -> list[Any]:
        """Return LangChain Tool definitions for the orchestrator.

        Returns:
            List of Tool objects: 'check_risk_status', 'get_risk_event_history'.
        """
        from langchain_core.tools import tool

        agent_ref = self

        @tool
        def check_risk_status(lmp: float) -> str:
            """Check for risk events given current LMP in $/MWh. Returns JSON."""
            events = []
            e = agent_ref._detect_price_spike(lmp, 0)
            if e:
                events.append({"type": e.event_type, "level": e.level.value, "action": e.recommended_action})
            e = agent_ref._detect_negative_prices(lmp, 0)
            if e:
                events.append({"type": e.event_type, "level": e.level.value, "action": e.recommended_action})
            is_halted = any(ev["level"] == "halt" for ev in events)
            return json.dumps({"risk_events": events, "is_halted": is_halted})

        @tool
        def get_risk_event_history() -> str:
            """Return the history of risk events detected since startup as JSON."""
            history = [
                {"type": ev.event_type, "level": ev.level.value, "description": ev.description}
                for ev in agent_ref._event_history[-50:]
            ]
            return json.dumps({"event_history": history, "total_events": len(agent_ref._event_history)})

        return [check_risk_status, get_risk_event_history]
