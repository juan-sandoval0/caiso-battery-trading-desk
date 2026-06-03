"""
Tests for BatteryDispatchOptimizer (Pyomo + HiGHS LP solver).

Tests verify:
    - LP solves to optimality on simple price scenarios
    - SoC bounds are never violated
    - Cycle limit is respected
    - DispatchConstraints from RiskMonitor are correctly applied
    - Naive and perfect-hindsight baselines produce plausible results

Note on price spread: the model charges a degradation cost of
``degradation_cost_per_kwh`` ($/kWh) on *both* the charge and discharge legs,
i.e. ~$100/MWh of round-trip throughput at the default $0.05/kWh. The fixture
therefore uses a wide spread (20 -> 300) so arbitrage is unambiguously
profitable and the behavioral assertions are robust to that penalty.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config.battery import BatteryConfig
from src.optimization.battery_dispatch import (
    BatteryDispatchOptimizer,
    DispatchConstraint,
    DispatchResult,
)


@pytest.fixture
def battery() -> BatteryConfig:
    """A small 2 MWh / 1 MW test battery."""
    return BatteryConfig(capacity_mwh=2.0, max_charge_mw=1.0, max_discharge_mw=1.0)


@pytest.fixture
def optimizer(battery: BatteryConfig) -> BatteryDispatchOptimizer:
    """Provide an optimizer with a small test battery."""
    return BatteryDispatchOptimizer(battery=battery)


@pytest.fixture
def simple_prices() -> np.ndarray:
    """24-hour price series: cheap overnight, expensive midday (hours 12–18)."""
    prices = np.full(24, 20.0)
    prices[12:18] = 300.0
    return prices


class TestSolveLP:
    """Tests for the main solve() method."""

    def test_returns_optimal_status(self, optimizer, simple_prices) -> None:
        """Solver returns an optimal/feasible status for a feasible price series."""
        result = optimizer.solve(simple_prices, interval_hours=1.0)
        assert isinstance(result, DispatchResult)
        assert any(s in result.status.lower() for s in ("optimal", "feasible"))

    def test_soc_never_below_minimum(self, optimizer, simple_prices, battery) -> None:
        """SoC trajectory stays above soc_min_mwh at all time steps."""
        result = optimizer.solve(simple_prices, interval_hours=1.0)
        assert np.all(result.soc_mwh >= battery.soc_min_mwh - 1e-6)

    def test_soc_never_above_maximum(self, optimizer, simple_prices, battery) -> None:
        """SoC trajectory stays below soc_max_mwh at all time steps."""
        result = optimizer.solve(simple_prices, interval_hours=1.0)
        assert np.all(result.soc_mwh <= battery.soc_max_mwh + 1e-6)

    def test_charges_during_cheap_hours(self, optimizer, simple_prices) -> None:
        """Optimizer charges during the low-price window and not during the peak."""
        result = optimizer.solve(simple_prices, interval_hours=1.0)
        # Total charging happens in cheap hours; none during the expensive window.
        assert result.charge_mw.sum() > 0.0
        assert result.charge_mw[12:18].sum() == pytest.approx(0.0, abs=1e-6)

    def test_discharges_during_expensive_hours(self, optimizer, simple_prices) -> None:
        """Optimizer discharges during the high-price window (hours 12–18)."""
        result = optimizer.solve(simple_prices, interval_hours=1.0)
        assert result.discharge_mw[12:18].sum() > 0.0

    def test_net_revenue_positive_for_spread(self, optimizer, simple_prices) -> None:
        """Net revenue is positive when price spread exceeds round-trip + degradation cost."""
        result = optimizer.solve(simple_prices, interval_hours=1.0)
        assert result.net_revenue_usd > 0.0

    def test_result_arrays_length_matches_horizon(self, optimizer, simple_prices) -> None:
        """charge_mw, discharge_mw, soc_mwh all have length equal to T."""
        result = optimizer.solve(simple_prices, interval_hours=1.0)
        T = len(simple_prices)
        assert len(result.charge_mw) == T
        assert len(result.discharge_mw) == T
        assert len(result.soc_mwh) == T

    def test_respects_cycle_limit(self, optimizer, simple_prices, battery) -> None:
        """Total discharged energy does not exceed max_cycles_per_day * capacity."""
        result = optimizer.solve(simple_prices, interval_hours=1.0)
        total_discharge_mwh = result.discharge_mw.sum() * 1.0
        limit = battery.max_cycles_per_day * battery.capacity_mwh
        assert total_discharge_mwh <= limit + 1e-6


class TestDispatchConstraints:
    """Tests that external DispatchConstraints are correctly enforced."""

    def test_halt_discharge_constraint(self, optimizer, simple_prices) -> None:
        """A discharge=0 constraint prevents discharge at the specified interval."""
        constraint = DispatchConstraint(12, "discharge", "eq", 0.0, "RiskMonitor")
        result = optimizer.solve(simple_prices, 1.0, constraints=[constraint])
        assert result.discharge_mw[12] == pytest.approx(0.0, abs=1e-6)

    def test_force_charge_constraint(self, optimizer, simple_prices, battery) -> None:
        """A charge=max constraint forces full charging at the specified interval."""
        # Force charge at hour 0 (already cheap, but assert it is exactly max).
        constraint = DispatchConstraint(0, "charge", "eq", battery.max_charge_mw, "RiskMonitor")
        result = optimizer.solve(simple_prices, 1.0, constraints=[constraint])
        assert result.charge_mw[0] == pytest.approx(battery.max_charge_mw, abs=1e-6)

    def test_multiple_constraints_applied(self, optimizer, simple_prices) -> None:
        """All constraints in a list are applied simultaneously."""
        constraints = [
            DispatchConstraint(12, "discharge", "eq", 0.0, "RiskMonitor"),
            DispatchConstraint(13, "discharge", "eq", 0.0, "RiskMonitor"),
        ]
        result = optimizer.solve(simple_prices, 1.0, constraints=constraints)
        assert result.discharge_mw[12] == pytest.approx(0.0, abs=1e-6)
        assert result.discharge_mw[13] == pytest.approx(0.0, abs=1e-6)


class TestBaselines:
    """Tests for baseline strategy methods."""

    def test_naive_baseline_returns_result(self, optimizer, simple_prices, battery) -> None:
        """solve_baseline_naive() returns a DispatchResult respecting SoC bounds."""
        result = optimizer.solve_baseline_naive(simple_prices, interval_hours=1.0)
        assert isinstance(result, DispatchResult)
        assert len(result.charge_mw) == len(simple_prices)
        assert np.all(result.soc_mwh >= battery.soc_min_mwh - 1e-6)
        assert np.all(result.soc_mwh <= battery.soc_max_mwh + 1e-6)

    def test_perfect_hindsight_revenue_gte_our_strategy(self, optimizer, simple_prices) -> None:
        """Perfect hindsight revenue is >= our strategy revenue (theoretical upper bound)."""
        # With a perfect forecast, solve() and perfect-hindsight should coincide;
        # perfect hindsight is never worse than dispatching on the same prices.
        ours = optimizer.solve(simple_prices, interval_hours=1.0)
        perfect = optimizer.solve_perfect_hindsight(simple_prices, interval_hours=1.0)
        assert perfect.net_revenue_usd >= ours.net_revenue_usd - 1e-6
