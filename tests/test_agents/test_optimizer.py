"""
Tests for DispatchOptimizerAgent.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.agents.optimizer import DispatchOptimizerAgent
from src.config.battery import DEFAULT_BATTERY


@pytest.fixture
def agent() -> DispatchOptimizerAgent:
    """Provide a DispatchOptimizerAgent with default battery config."""
    raise NotImplementedError(
        # return DispatchOptimizerAgent(battery=DEFAULT_BATTERY)
    )


class TestRunBacktestDay:
    """Tests for run_backtest_day()."""

    def test_revenue_within_perfect_hindsight_bounds(self, agent) -> None:
        """Actual revenue is <= perfect hindsight revenue."""
        raise NotImplementedError()

    def test_returns_expected_keys(self, agent) -> None:
        """run_backtest_day() returns dict with all required metric keys."""
        raise NotImplementedError()
