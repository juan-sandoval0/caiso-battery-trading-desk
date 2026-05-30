"""
Tests for ConflictResolver and TradingDeskCoordinator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.risk_monitor import RiskEvent, RiskLevel
from src.optimization.battery_dispatch import DispatchConstraint, DispatchResult
from src.orchestrator.conflict_resolver import (
    apply_constraint_to_schedule,
    merge_constraints,
    resolve_conflicts,
)


def _make_result(T: int = 24, revenue: float = 100.0) -> DispatchResult:
    """Build a synthetic DispatchResult for testing.

    Args:
        T: Number of time intervals.
        revenue: Net revenue value.

    Returns:
        DispatchResult with uniform 0.5 MW charge and 0.5 MW discharge.
    """
    charge = np.full(T, 0.5)
    discharge = np.full(T, 0.5)
    soc = np.linspace(2.0, 2.0, T)
    return DispatchResult(charge, discharge, soc, revenue, 0.01, "optimal")


class TestResolveConflicts:
    """Tests for resolve_conflicts()."""

    def test_halt_zeros_entire_schedule(self) -> None:
        """When is_halted=True, all charge and discharge values become zero."""
        result = _make_result()
        resolved = resolve_conflicts(result, [], is_halted=True)
        assert np.all(resolved.charge_mw == 0.0)
        assert np.all(resolved.discharge_mw == 0.0)
        assert resolved.status == "HALTED"
        assert resolved.net_revenue_usd == 0.0

    def test_halt_preserves_initial_soc(self) -> None:
        """HALT keeps SoC flat at the initial value."""
        result = _make_result()
        initial_soc = float(result.soc_mwh[0])
        resolved = resolve_conflicts(result, [], is_halted=True)
        assert np.all(resolved.soc_mwh == initial_soc)

    def test_no_halt_no_events_returns_unchanged(self) -> None:
        """When is_halted=False and no events, schedule is returned unchanged."""
        result = _make_result()
        resolved = resolve_conflicts(result, [], is_halted=False)
        assert resolved is result  # exact same object when nothing to do

    def test_no_halt_preserves_schedule_values(self) -> None:
        """When is_halted=False and no events, charge/discharge arrays are unchanged."""
        result = _make_result()
        resolved = resolve_conflicts(result, [], is_halted=False)
        np.testing.assert_array_equal(resolved.charge_mw, result.charge_mw)
        np.testing.assert_array_equal(resolved.discharge_mw, result.discharge_mw)

    def test_constrain_event_modifies_interval(self) -> None:
        """A CONSTRAIN risk event modifies the correct interval in the schedule."""
        result = _make_result(T=24)
        con = DispatchConstraint(
            interval_idx=5,
            variable="discharge",
            operator="eq",
            value_mw=0.0,
            source="RiskMonitor",
        )
        event = RiskEvent(
            event_type="PRICE_SPIKE",
            level=RiskLevel.CRITICAL,
            description="spike",
            recommended_action="CONSTRAIN",
            constraints=[con],
        )
        resolved = resolve_conflicts(result, [event], is_halted=False)
        assert resolved.discharge_mw[5] == 0.0
        # Other intervals should be unchanged
        assert resolved.discharge_mw[0] == pytest.approx(0.5)
        assert resolved.status.endswith("+risk_adjusted")

    def test_non_constrain_events_not_applied(self) -> None:
        """ALERT-only events do not modify the schedule."""
        result = _make_result()
        event = RiskEvent(
            event_type="RAMP_EVENT",
            level=RiskLevel.WARNING,
            description="ramp",
            recommended_action="ALERT",
            constraints=[],
        )
        resolved = resolve_conflicts(result, [event], is_halted=False)
        # ALERT event has no constraints → schedule unchanged
        np.testing.assert_array_equal(resolved.charge_mw, result.charge_mw)


class TestMergeConstraints:
    """Tests for merge_constraints() priority ordering."""

    def test_risk_constraint_overrides_optimizer(self) -> None:
        """Risk constraint wins when it conflicts with an optimizer constraint."""
        opt_con = DispatchConstraint(3, "discharge", "le", 0.8, "optimizer")
        risk_con = DispatchConstraint(3, "discharge", "eq", 0.0, "RiskMonitor")
        merged = merge_constraints([opt_con], [risk_con], [])
        # Only one entry for (interval=3, variable=discharge)
        assert len(merged) == 1
        assert merged[0].value_mw == 0.0
        assert merged[0].source == "RiskMonitor"

    def test_intel_constraint_overrides_optimizer(self) -> None:
        """Intel constraint beats optimizer for the same key."""
        opt_con = DispatchConstraint(2, "charge", "le", 0.5, "optimizer")
        intel_con = DispatchConstraint(2, "charge", "le", 0.3, "MarketIntel")
        merged = merge_constraints([opt_con], [], [intel_con])
        assert len(merged) == 1
        assert merged[0].value_mw == pytest.approx(0.3)

    def test_risk_overrides_intel(self) -> None:
        """Risk constraint has highest priority."""
        intel_con = DispatchConstraint(1, "discharge", "le", 0.5, "MarketIntel")
        risk_con = DispatchConstraint(1, "discharge", "eq", 0.0, "RiskMonitor")
        merged = merge_constraints([], [risk_con], [intel_con])
        assert len(merged) == 1
        assert merged[0].source == "RiskMonitor"

    def test_non_conflicting_constraints_all_present(self) -> None:
        """Constraints targeting different intervals are all preserved."""
        opt_con = DispatchConstraint(0, "charge", "le", 1.0, "optimizer")
        risk_con = DispatchConstraint(5, "discharge", "eq", 0.0, "RiskMonitor")
        intel_con = DispatchConstraint(10, "charge", "ge", 0.2, "MarketIntel")
        merged = merge_constraints([opt_con], [risk_con], [intel_con])
        assert len(merged) == 3


class TestApplyConstraint:
    """Tests for apply_constraint_to_schedule()."""

    def test_eq_sets_exact_value(self) -> None:
        """'eq' operator sets the array value to exactly the constraint value."""
        c = np.full(10, 0.5)
        d = np.full(10, 0.5)
        con = DispatchConstraint(3, "discharge", "eq", 0.0, "test")
        new_c, new_d = apply_constraint_to_schedule(con, c, d)
        assert new_d[3] == pytest.approx(0.0)

    def test_le_clips_to_upper_bound(self) -> None:
        """'le' operator clips the array value to at most the constraint value."""
        c = np.full(10, 0.8)
        d = np.full(10, 0.8)
        con = DispatchConstraint(4, "charge", "le", 0.3, "test")
        new_c, new_d = apply_constraint_to_schedule(con, c, d)
        assert new_c[4] == pytest.approx(0.3)

    def test_le_does_not_increase_below_bound(self) -> None:
        """'le' with a value above current leaves array unchanged."""
        c = np.full(10, 0.2)
        d = np.full(10, 0.5)
        con = DispatchConstraint(2, "charge", "le", 0.9, "test")
        new_c, _ = apply_constraint_to_schedule(con, c, d)
        assert new_c[2] == pytest.approx(0.2)

    def test_ge_clips_to_lower_bound(self) -> None:
        """'ge' operator clips the array value to at least the constraint value."""
        c = np.full(10, 0.1)
        d = np.full(10, 0.5)
        con = DispatchConstraint(1, "charge", "ge", 0.5, "test")
        new_c, _ = apply_constraint_to_schedule(con, c, d)
        assert new_c[1] == pytest.approx(0.5)

    def test_original_arrays_not_mutated(self) -> None:
        """The input arrays are not modified in place."""
        c = np.full(10, 0.5)
        d = np.full(10, 0.5)
        c_orig = c.copy()
        d_orig = d.copy()
        con = DispatchConstraint(0, "discharge", "eq", 0.0, "test")
        apply_constraint_to_schedule(con, c, d)
        np.testing.assert_array_equal(c, c_orig)
        np.testing.assert_array_equal(d, d_orig)

    def test_out_of_bounds_interval_ignored(self) -> None:
        """Constraint with interval_idx >= T is ignored without error."""
        c = np.full(5, 0.5)
        d = np.full(5, 0.5)
        con = DispatchConstraint(10, "discharge", "eq", 0.0, "test")
        new_c, new_d = apply_constraint_to_schedule(con, c, d)
        np.testing.assert_array_equal(new_c, c)
        np.testing.assert_array_equal(new_d, d)
