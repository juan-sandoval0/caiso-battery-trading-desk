"""
Tests for DispatchOptimizerAgent.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.optimizer import DispatchOptimizerAgent
from src.config.battery import DEFAULT_BATTERY


@pytest.fixture
def agent() -> DispatchOptimizerAgent:
    """Provide a DispatchOptimizerAgent with default battery config."""
    return DispatchOptimizerAgent(battery=DEFAULT_BATTERY)


@pytest.fixture
def spread_prices() -> np.ndarray:
    """24h prices with a wide, clearly-arbitrageable spread."""
    prices = np.full(24, 20.0)
    prices[12:18] = 300.0
    return prices


class TestRunBacktestDay:
    """Tests for run_backtest_day()."""

    def test_revenue_within_perfect_hindsight_bounds(self, agent, spread_prices) -> None:
        """Actual revenue is <= perfect hindsight revenue."""
        result = agent.run_backtest_day(
            date="2024-07-01",
            actual_prices=spread_prices,
            forecast_prices=spread_prices,  # perfect forecast
            initial_soc_mwh=2.0,
        )
        assert result["revenue_usd"] <= result["perfect_revenue_usd"] + 1e-6

    def test_returns_expected_keys(self, agent, spread_prices) -> None:
        """run_backtest_day() returns dict with all required metric keys."""
        result = agent.run_backtest_day(
            date="2024-07-01",
            actual_prices=spread_prices,
            forecast_prices=spread_prices,
            initial_soc_mwh=2.0,
        )
        required = {
            "date",
            "revenue_usd",
            "cycles",
            "final_soc_mwh",
            "vs_naive_usd",
            "vs_perfect_usd",
            "forecast_rmse",
            "solver_status",
        }
        assert required <= set(result)
        assert result["forecast_rmse"] == pytest.approx(0.0, abs=1e-9)
