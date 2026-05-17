"""
DispatchOptimizerAgent — battery dispatch scheduling agent.

Wraps the Pyomo LP solver and translates between the orchestrator's
SharedMarketState (price forecasts + external constraints) and the
BatteryDispatchOptimizer's internal representation.

Responsibilities:
    - Receive price forecast from SharedMarketState (produced by PriceForecasterAgent)
    - Collect any active DispatchConstraints from RiskMonitor / MarketIntelAgent
    - Invoke BatteryDispatchOptimizer.solve() and return DispatchResult
    - Write the resulting schedule to SharedMarketState
    - Log each decision to MarketDB for audit and dashboard display

This agent is the DECISION MAKER for physical dispatch but can be overridden
by the RiskMonitorAgent through the ConflictResolver.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config.battery import BatteryConfig, DEFAULT_BATTERY
from src.config.nodes import DEFAULT_HUB_NODE
from src.optimization.battery_dispatch import (
    BatteryDispatchOptimizer,
    DispatchConstraint,
    DispatchResult,
)


class DispatchOptimizerAgent:
    """Produces battery charge/discharge schedules by solving the dispatch LP.

    Usage::

        agent = DispatchOptimizerAgent(battery=DEFAULT_BATTERY)
        result = agent.run(state=shared_state, db=market_db)
    """

    def __init__(
        self,
        battery: BatteryConfig = DEFAULT_BATTERY,
        node: str = DEFAULT_HUB_NODE,
    ) -> None:
        """Initialize the agent.

        Args:
            battery: BESS physical/financial parameters.
            node: CAISO hub node this agent optimizes for.
        """
        raise NotImplementedError(
            # Store self._battery = battery, self._node = node.
            # Instantiate self._optimizer = BatteryDispatchOptimizer(battery=battery).
            # Initialize self._current_soc_mwh = battery.initial_soc_pct * battery.capacity_mwh.
        )

    def run(self, state: Any, db: Any) -> DispatchResult:
        """Execute one optimization tick using the current SharedMarketState.

        Args:
            state: SharedMarketState with price forecast and constraints.
            db: MarketDB instance for logging the decision.

        Returns:
            DispatchResult with the optimal charge/discharge schedule.
        """
        raise NotImplementedError(
            # 1. Extract price_forecast.da_lmp_forecast from state.
            # 2. Extract active DispatchConstraints from state.active_constraints.
            # 3. Call self._optimizer.solve(prices, interval_hours=1.0,
            #       initial_soc_mwh=self._current_soc_mwh, constraints=constraints).
            # 4. Update self._current_soc_mwh = result.soc_mwh[-1].
            # 5. Log decision to db.log_agent_decision().
            # 6. Return result.
        )

    async def arun(self, state: Any, db: Any) -> DispatchResult:
        """Async version of run() for the LangGraph async graph.

        Args:
            state: SharedMarketState.
            db: MarketDB instance.

        Returns:
            DispatchResult.
        """
        raise NotImplementedError(
            # Wrap self.run() in asyncio.get_event_loop().run_in_executor()
            # since Pyomo solve is CPU-bound / blocking.
        )

    def run_backtest_day(
        self,
        date: str,
        actual_prices: np.ndarray,
        forecast_prices: np.ndarray,
        initial_soc_mwh: float | None = None,
    ) -> dict[str, float]:
        """Run one day of backtest and return P&L metrics.

        Args:
            date: ISO date string (for logging).
            actual_prices: Realized LMP values used for revenue calculation.
            forecast_prices: Model forecast prices used for scheduling.
            initial_soc_mwh: Override starting SoC (defaults to instance state).

        Returns:
            Dict with keys: 'revenue_usd', 'cycles', 'final_soc_mwh',
            'vs_naive_usd', 'vs_perfect_usd'.
        """
        raise NotImplementedError(
            # 1. Solve with forecast_prices to get schedule.
            # 2. Compute actual revenue by applying schedule to actual_prices.
            # 3. Solve baseline naive (solve_baseline_naive) and perfect hindsight.
            # 4. Return comparison dict.
        )

    def get_langchain_tools(self) -> list[Any]:
        """Return LangChain Tool definitions for the orchestrator.

        Returns:
            List of Tool objects: 'run_dispatch_optimization', 'get_current_soc'.
        """
        raise NotImplementedError(
            # Define tools that the LangGraph supervisor can invoke.
        )
