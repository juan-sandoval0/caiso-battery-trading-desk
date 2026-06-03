"""
Tests for RiskMonitorAgent.

All tests use synthetic state objects — no database or CAISO API access required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.agents.risk_monitor import RiskLevel, RiskMonitorAgent


@pytest.fixture
def monitor() -> RiskMonitorAgent:
    """Provide a RiskMonitorAgent with default thresholds."""
    return RiskMonitorAgent()


def _make_state(lmp: float = 50.0, intel_text: str = "") -> MagicMock:
    """Build a minimal mock SharedMarketState for testing.

    Args:
        lmp: Latest RT LMP value.
        intel_text: Market intelligence text to inject.

    Returns:
        MagicMock with the attributes run() reads.
    """
    state = MagicMock()
    state.latest_rt_lmp = lmp
    # Single-row frame => ramp detection (which needs >1 row) is skipped.
    state.recent_lmp_df = pd.DataFrame({"lmp": [lmp]})
    state.dispatch_result = None
    if intel_text:
        intel = MagicMock()
        intel.raw_summary = intel_text
        state.market_intel = intel
    else:
        state.market_intel = None
    return state


class TestPriceSpikeDetection:
    """Tests for _detect_price_spike()."""

    def test_spike_detected_above_threshold(self, monitor) -> None:
        """Returns a CRITICAL RiskEvent when LMP > 500."""
        event = monitor._detect_price_spike(600.0, interval_idx=0)
        assert event is not None
        assert event.level == RiskLevel.CRITICAL
        assert event.event_type == "PRICE_SPIKE"

    def test_no_spike_below_threshold(self, monitor) -> None:
        """Returns None when LMP is within normal range."""
        assert monitor._detect_price_spike(100.0, interval_idx=0) is None

    def test_spike_event_constrains_discharge(self, monitor) -> None:
        """Spike event includes a discharge=0 DispatchConstraint."""
        event = monitor._detect_price_spike(600.0, interval_idx=3)
        assert event.recommended_action == "CONSTRAIN"
        assert len(event.constraints) == 1
        c = event.constraints[0]
        assert c.variable == "discharge"
        assert c.operator == "eq"
        assert c.value_mw == pytest.approx(0.0)
        assert c.interval_idx == 3


class TestNegativePriceDetection:
    """Tests for _detect_negative_prices()."""

    def test_negative_price_detected(self, monitor) -> None:
        """Returns a RiskEvent when LMP < -50."""
        event = monitor._detect_negative_prices(-75.0, interval_idx=0)
        assert event is not None
        assert event.event_type == "NEGATIVE_PRICES"

    def test_negative_price_event_forces_charging(self, monitor) -> None:
        """Negative price event includes a charge=max DispatchConstraint."""
        event = monitor._detect_negative_prices(-75.0, interval_idx=0)
        assert event.recommended_action == "CONSTRAIN"
        assert len(event.constraints) == 1
        c = event.constraints[0]
        assert c.variable == "charge"
        assert c.operator == "eq"
        assert c.value_mw > 0.0


class TestEEADetection:
    """Tests for _detect_eea()."""

    def test_eea3_triggers_halt(self, monitor) -> None:
        """Text containing 'EEA3' produces a HALT-level RiskEvent."""
        event = monitor._detect_eea("CAISO has declared EEA3 across the grid")
        assert event is not None
        assert event.level == RiskLevel.HALT
        assert event.recommended_action == "HALT"

    def test_eea1_triggers_warning_not_halt(self, monitor) -> None:
        """Text containing 'EEA1' produces a non-HALT event (implementation: CRITICAL/CONSTRAIN)."""
        event = monitor._detect_eea("EEA1 watch issued for this afternoon")
        assert event is not None
        assert event.level != RiskLevel.HALT
        assert event.recommended_action != "HALT"

    def test_no_eea_keywords_returns_none(self, monitor) -> None:
        """Clean market intel text returns None."""
        assert monitor._detect_eea("Mild weather, no grid stress expected.") is None


class TestRunMethod:
    """Integration-style tests for the full run() method."""

    def test_run_returns_empty_list_for_normal_conditions(self, monitor) -> None:
        """run() returns an empty list when all metrics are within safe bounds."""
        state = _make_state(lmp=45.0, intel_text="")
        events = monitor.run(state, db=MagicMock())
        assert events == []

    def test_run_returns_events_for_spike(self, monitor) -> None:
        """run() returns at least one event when LMP is above spike threshold."""
        state = _make_state(lmp=750.0, intel_text="")
        events = monitor.run(state, db=MagicMock())
        assert len(events) >= 1
        assert any(e.event_type == "PRICE_SPIKE" for e in events)
