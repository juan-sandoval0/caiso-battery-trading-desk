"""
Tests for SharedMarketState.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.orchestrator.shared_state import SharedMarketState


@pytest.fixture
def base_state() -> SharedMarketState:
    """Provide a minimal SharedMarketState for testing."""
    raise NotImplementedError(
        # return SharedMarketState(tick_time=datetime.now(), node='TH_NP15_GEN-APND')
    )


class TestImmutability:
    """Tests that state mutations return new instances."""

    def test_with_price_forecast_returns_new_instance(self, base_state) -> None:
        """with_price_forecast() returns a new object, not the same instance."""
        raise NotImplementedError()

    def test_original_state_unchanged_after_with_call(self, base_state) -> None:
        """Calling with_* methods does not modify the original state object."""
        raise NotImplementedError()


class TestSerialization:
    """Tests for to_dict() / from_dict() roundtrip."""

    def test_roundtrip_preserves_scalar_fields(self, base_state) -> None:
        """Scalar fields survive to_dict() → from_dict() roundtrip unchanged."""
        raise NotImplementedError()

    def test_roundtrip_with_price_forecast(self, base_state) -> None:
        """PriceForecast survives serialization roundtrip."""
        raise NotImplementedError()


class TestConflictMerge:
    """Tests for constraint merging in with_risk_events()."""

    def test_halt_event_sets_is_halted(self, base_state) -> None:
        """A HALT-level RiskEvent sets state.is_halted = True."""
        raise NotImplementedError()

    def test_constraints_appended_from_events(self, base_state) -> None:
        """DispatchConstraints from risk events are added to active_constraints."""
        raise NotImplementedError()
