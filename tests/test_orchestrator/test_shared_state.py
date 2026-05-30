"""
Tests for SharedMarketState.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.agents.forecaster import PriceForecast
from src.agents.market_intel import MarketIntelSummary
from src.agents.risk_monitor import RiskEvent, RiskLevel
from src.optimization.battery_dispatch import DispatchConstraint, DispatchResult
from src.orchestrator.shared_state import SharedMarketState


@pytest.fixture
def base_state() -> SharedMarketState:
    """Provide a minimal SharedMarketState for testing."""
    return SharedMarketState(tick_time=datetime(2024, 6, 15, 12, 0, 0), node="TH_NP15_GEN-APND")


@pytest.fixture
def sample_forecast() -> PriceForecast:
    """Provide a synthetic PriceForecast."""
    times = pd.date_range("2024-06-15 13:00", periods=24, freq="h")
    da = np.linspace(40.0, 120.0, 24)
    return PriceForecast(
        node="TH_NP15_GEN-APND",
        times=times,
        da_lmp_forecast=da,
        rt_lmp_forecast=None,
        confidence_low=None,
        confidence_high=None,
        model_version="test_v1",
    )


@pytest.fixture
def sample_dispatch() -> DispatchResult:
    """Provide a synthetic DispatchResult."""
    T = 24
    return DispatchResult(
        charge_mw=np.full(T, 0.5),
        discharge_mw=np.full(T, 0.5),
        soc_mwh=np.linspace(2.0, 2.0, T),
        net_revenue_usd=150.0,
        solve_time_s=0.05,
        status="optimal",
    )


class TestImmutability:
    """Tests that state mutations return new instances."""

    def test_with_price_forecast_returns_new_instance(
        self, base_state: SharedMarketState, sample_forecast: PriceForecast
    ) -> None:
        """with_price_forecast() returns a new object, not the same instance."""
        new_state = base_state.with_price_forecast(sample_forecast)
        assert new_state is not base_state
        assert new_state.price_forecast is sample_forecast

    def test_original_state_unchanged_after_with_call(
        self, base_state: SharedMarketState, sample_forecast: PriceForecast
    ) -> None:
        """Calling with_* methods does not modify the original state object."""
        _ = base_state.with_price_forecast(sample_forecast)
        assert base_state.price_forecast is None

    def test_with_dispatch_result_returns_new_instance(
        self, base_state: SharedMarketState, sample_dispatch: DispatchResult
    ) -> None:
        new_state = base_state.with_dispatch_result(sample_dispatch)
        assert new_state is not base_state
        assert new_state.dispatch_result is sample_dispatch

    def test_with_market_intel_returns_new_instance(self, base_state: SharedMarketState) -> None:
        intel = MarketIntelSummary(raw_summary="no events today")
        new_state = base_state.with_market_intel(intel)
        assert new_state is not base_state
        assert new_state.market_intel is intel


class TestSerialization:
    """Tests for to_dict() / from_dict() roundtrip."""

    def test_roundtrip_preserves_scalar_fields(self, base_state: SharedMarketState) -> None:
        """Scalar fields survive to_dict() → from_dict() roundtrip unchanged."""
        state = SharedMarketState(
            tick_time=datetime(2024, 6, 15, 12, 0, 0),
            node="TH_NP15_GEN-APND",
            latest_rt_lmp=85.5,
            current_soc_mwh=3.2,
            cumulative_pnl_usd=1234.56,
            is_halted=False,
        )
        d = state.to_dict()
        restored = SharedMarketState.from_dict(d)

        assert restored.node == state.node
        assert abs(restored.latest_rt_lmp - state.latest_rt_lmp) < 1e-9
        assert abs(restored.current_soc_mwh - state.current_soc_mwh) < 1e-9
        assert abs(restored.cumulative_pnl_usd - state.cumulative_pnl_usd) < 1e-9
        assert restored.is_halted == state.is_halted

    def test_roundtrip_with_price_forecast(
        self, base_state: SharedMarketState, sample_forecast: PriceForecast
    ) -> None:
        """PriceForecast survives serialization roundtrip."""
        state = base_state.with_price_forecast(sample_forecast)
        d = state.to_dict()
        restored = SharedMarketState.from_dict(d)

        assert restored.price_forecast is not None
        np.testing.assert_allclose(
            restored.price_forecast.da_lmp_forecast,
            sample_forecast.da_lmp_forecast,
            rtol=1e-9,
        )
        assert restored.price_forecast.node == sample_forecast.node
        assert restored.price_forecast.model_version == sample_forecast.model_version

    def test_roundtrip_with_dispatch_result(
        self, base_state: SharedMarketState, sample_dispatch: DispatchResult
    ) -> None:
        """DispatchResult survives serialization roundtrip."""
        state = base_state.with_dispatch_result(sample_dispatch)
        d = state.to_dict()
        restored = SharedMarketState.from_dict(d)

        assert restored.dispatch_result is not None
        np.testing.assert_allclose(
            restored.dispatch_result.charge_mw, sample_dispatch.charge_mw, rtol=1e-9
        )
        assert restored.dispatch_result.status == sample_dispatch.status
        assert abs(restored.dispatch_result.net_revenue_usd - sample_dispatch.net_revenue_usd) < 1e-9

    def test_roundtrip_none_fields(self) -> None:
        """State with all optional fields None roundtrips cleanly."""
        state = SharedMarketState(tick_time=datetime(2024, 1, 1), node="TH_SP15_GEN-APND")
        d = state.to_dict()
        restored = SharedMarketState.from_dict(d)
        assert restored.price_forecast is None
        assert restored.dispatch_result is None
        assert restored.market_intel is None
        assert restored.recent_lmp_df is None


class TestConflictMerge:
    """Tests for constraint merging in with_risk_events()."""

    def test_halt_event_sets_is_halted(self, base_state: SharedMarketState) -> None:
        """A HALT-level RiskEvent sets state.is_halted = True."""
        halt_event = RiskEvent(
            event_type="EEA3",
            level=RiskLevel.HALT,
            description="EEA3 detected",
            recommended_action="HALT",
        )
        new_state = base_state.with_risk_events([halt_event])
        assert new_state.is_halted is True

    def test_warning_event_does_not_halt(self, base_state: SharedMarketState) -> None:
        """A WARNING-level event does not set is_halted."""
        warn_event = RiskEvent(
            event_type="RAMP_EVENT",
            level=RiskLevel.WARNING,
            description="ramp detected",
            recommended_action="ALERT",
        )
        new_state = base_state.with_risk_events([warn_event])
        assert new_state.is_halted is False

    def test_constraints_appended_from_events(self, base_state: SharedMarketState) -> None:
        """DispatchConstraints from risk events are added to active_constraints."""
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
        new_state = base_state.with_risk_events([event])
        assert len(new_state.active_constraints) == 1
        assert new_state.active_constraints[0].interval_idx == 5

    def test_constraints_accumulate_across_calls(self, base_state: SharedMarketState) -> None:
        """Multiple with_risk_events() calls accumulate constraints."""
        con1 = DispatchConstraint(0, "discharge", "eq", 0.0, "RiskMonitor")
        con2 = DispatchConstraint(1, "charge", "eq", 1.0, "RiskMonitor")
        event1 = RiskEvent("E1", RiskLevel.CRITICAL, "e1", "CONSTRAIN", constraints=[con1])
        event2 = RiskEvent("E2", RiskLevel.WARNING, "e2", "CONSTRAIN", constraints=[con2])

        state1 = base_state.with_risk_events([event1])
        state2 = state1.with_risk_events([event2])
        assert len(state2.active_constraints) == 2
