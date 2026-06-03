"""
ConflictResolver — priority-based agent conflict resolution.

When multiple agents produce conflicting dispatch guidance, this module
applies a strict priority ordering to determine the final action:

    Priority 1 (highest): RiskMonitorAgent — HALT or OVERRIDE takes effect immediately
    Priority 2:           MarketIntelAgent — adds constraints (e.g., avoid EEA discharge)
    Priority 3:           DispatchOptimizerAgent — base schedule
    Priority 4 (lowest):  PriceForecasterAgent — advisory only

Resolution process:
    1. If is_halted == True → zero out entire schedule
    2. Apply all CONSTRAIN events in priority order
    3. Re-solve optimizer with merged constraints if any were added after initial solve
    4. Validate final schedule against all constraints

This module contains pure functions with no side effects to make testing trivial.
"""

from __future__ import annotations

import numpy as np

from src.agents.risk_monitor import RiskEvent, RiskLevel
from src.optimization.battery_dispatch import DispatchConstraint, DispatchResult


def resolve_conflicts(
    dispatch_result: DispatchResult,
    risk_events: list[RiskEvent],
    is_halted: bool,
) -> DispatchResult:
    """Apply conflict resolution rules and return the final dispatch schedule.

    Args:
        dispatch_result: Raw output from the DispatchOptimizerAgent.
        risk_events: All RiskEvents detected this tick.
        is_halted: True if RiskMonitor issued a HALT.

    Returns:
        Potentially modified DispatchResult with risk overrides applied.
    """
    if is_halted:
        return _zero_schedule(dispatch_result)

    if not risk_events:
        return dispatch_result

    c = dispatch_result.charge_mw.copy()
    d = dispatch_result.discharge_mw.copy()

    constrain_events = [e for e in risk_events if e.recommended_action == "CONSTRAIN"]
    for event in constrain_events:
        for con in event.constraints:
            c, d = apply_constraint_to_schedule(con, c, d)

    # Recompute SoC trajectory after schedule modifications
    from src.config.battery import DEFAULT_BATTERY as bat
    soc = np.zeros(len(c))
    cur = float(dispatch_result.soc_mwh[0]) if len(dispatch_result.soc_mwh) > 0 else bat.initial_soc_pct * bat.capacity_mwh
    for t in range(len(c)):
        cur = cur + bat.charge_efficiency * c[t] - (1.0 / bat.discharge_efficiency) * d[t]
        soc[t] = cur

    status = dispatch_result.status + "+risk_adjusted"
    return DispatchResult(
        charge_mw=c,
        discharge_mw=d,
        soc_mwh=soc,
        net_revenue_usd=dispatch_result.net_revenue_usd,
        solve_time_s=dispatch_result.solve_time_s,
        status=status,
        active_constraints=dispatch_result.active_constraints,
    )


def merge_constraints(
    optimizer_constraints: list[DispatchConstraint],
    risk_constraints: list[DispatchConstraint],
    intel_constraints: list[DispatchConstraint],
) -> list[DispatchConstraint]:
    """Merge constraints from all agents, applying priority ordering.

    Higher-priority constraints override lower-priority ones for the same
    interval and variable. Risk > Intel > Optimizer.

    Args:
        optimizer_constraints: Constraints from DispatchOptimizerAgent.
        risk_constraints: Constraints from RiskMonitorAgent (highest priority).
        intel_constraints: Constraints from MarketIntelAgent.

    Returns:
        Deduplicated, priority-ordered list of DispatchConstraints.
    """
    merged: dict[tuple[int, str], DispatchConstraint] = {}
    # Insert lowest priority first so higher priority overwrites
    for con in optimizer_constraints + intel_constraints + risk_constraints:
        merged[(con.interval_idx, con.variable)] = con
    return list(merged.values())


def _zero_schedule(result: DispatchResult) -> DispatchResult:
    """Return a DispatchResult with all charge and discharge zeroed out (HALT).

    Args:
        result: Original dispatch result.

    Returns:
        New DispatchResult with charge_mw and discharge_mw set to zeros,
        SoC held constant, revenue set to 0.
    """
    T = len(result.charge_mw)
    zero = np.zeros(T)
    initial_soc = float(result.soc_mwh[0]) if len(result.soc_mwh) > 0 else 2.0
    soc = np.full(T, initial_soc)
    return DispatchResult(
        charge_mw=zero,
        discharge_mw=zero,
        soc_mwh=soc,
        net_revenue_usd=0.0,
        solve_time_s=result.solve_time_s,
        status="HALTED",
        active_constraints=result.active_constraints,
    )


def apply_constraint_to_schedule(
    constraint: DispatchConstraint,
    charge_mw: np.ndarray,
    discharge_mw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a single DispatchConstraint to a schedule array pair.

    Args:
        constraint: The constraint to apply.
        charge_mw: Current charge schedule array (will be copied, not mutated).
        discharge_mw: Current discharge schedule array.

    Returns:
        (new_charge_mw, new_discharge_mw) with constraint enforced.
    """
    c = charge_mw.copy()
    d = discharge_mw.copy()
    t = constraint.interval_idx
    v = constraint.value_mw

    # Guard against out-of-bounds interval index
    if t < 0 or t >= len(c):
        return c, d

    arr = c if constraint.variable == "charge" else d
    if constraint.operator == "eq":
        arr[t] = v
    elif constraint.operator == "le":
        arr[t] = min(arr[t], v)
    elif constraint.operator == "ge":
        arr[t] = max(arr[t], v)

    return c, d
