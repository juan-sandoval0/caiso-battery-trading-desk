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
    raise NotImplementedError(
        # If is_halted: return _zero_schedule(dispatch_result).
        # Collect all constraints from risk_events where level >= WARNING.
        # Apply each constraint to the schedule arrays (modify charge/discharge in-place copy).
        # Recompute soc_mwh trajectory after modifications.
        # Return new DispatchResult with modified arrays and a note in status.
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
    raise NotImplementedError(
        # Build a dict keyed by (interval_idx, variable).
        # Insert in priority order: optimizer first (lowest), then intel, then risk.
        # Later writes override earlier ones (risk wins ties).
        # Return values of the dict as a list.
    )


def _zero_schedule(result: DispatchResult) -> DispatchResult:
    """Return a DispatchResult with all charge and discharge zeroed out (HALT).

    Args:
        result: Original dispatch result.

    Returns:
        New DispatchResult with charge_mw and discharge_mw set to zeros,
        SoC held constant, revenue set to 0.
    """
    raise NotImplementedError(
        # T = len(result.charge_mw)
        # zero = np.zeros(T)
        # soc = np.full(T, result.soc_mwh[0])  # SoC stays flat
        # return DispatchResult(zero, zero, soc, 0.0, result.solve_time_s, 'HALTED',
        #   active_constraints=result.active_constraints)
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
    raise NotImplementedError(
        # Copy arrays. Apply constraint.operator ('eq', 'le', 'ge') to
        # constraint.variable ('charge', 'discharge') at constraint.interval_idx.
        # Return modified copies.
    )
