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

import asyncio
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
        self._battery = battery
        self._node = node
        self._optimizer = BatteryDispatchOptimizer(battery=battery)
        self._current_soc_mwh: float = battery.initial_soc_pct * battery.capacity_mwh

    def run(self, state: Any, db: Any) -> DispatchResult:
        """Execute one optimization tick using the current SharedMarketState.

        Args:
            state: SharedMarketState with price forecast and constraints.
            db: MarketDB instance for logging the decision.

        Returns:
            DispatchResult with the optimal charge/discharge schedule.
        """
        forecast = state.price_forecast
        prices = forecast.da_lmp_forecast

        external_constraints: list[DispatchConstraint] = getattr(
            state, "active_constraints", []
        ) or []

        result = self._optimizer.solve(
            prices_mwh=prices,
            interval_hours=1.0,
            initial_soc_mwh=self._current_soc_mwh,
            constraints=external_constraints,
        )

        self._current_soc_mwh = float(result.soc_mwh[-1])

        db.log_agent_decision(
            agent="DispatchOptimizer",
            action="DISPATCH_SCHEDULE",
            rationale=f"LP solved ({result.status}), "
                      f"net_revenue=${result.net_revenue_usd:.2f}, "
                      f"solve_time={result.solve_time_s:.3f}s",
            metadata={
                "status": result.status,
                "net_revenue_usd": result.net_revenue_usd,
                "solve_time_s": result.solve_time_s,
                "final_soc_mwh": float(result.soc_mwh[-1]),
                "n_constraints": len(external_constraints),
            },
        )

        return result

    async def arun(self, state: Any, db: Any) -> DispatchResult:
        """Async version of run() for the LangGraph async graph.

        Pyomo solve is CPU-bound and blocking, so it runs in a thread executor.

        Args:
            state: SharedMarketState.
            db: MarketDB instance.

        Returns:
            DispatchResult.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, state, db)

    def run_backtest_day(
        self,
        date: str,
        actual_prices: np.ndarray,
        forecast_prices: np.ndarray,
        initial_soc_mwh: float | None = None,
    ) -> dict[str, float]:
        """Run one day of backtest and return P&L metrics.

        Solves dispatch with forecast_prices, then evaluates revenue against
        actual_prices. Also computes naive and perfect-hindsight baselines.

        Args:
            date: ISO date string (for logging).
            actual_prices: Realized LMP values used for revenue calculation.
            forecast_prices: Model forecast prices used for scheduling.
            initial_soc_mwh: Override starting SoC (defaults to instance state).

        Returns:
            Dict with keys: 'date', 'revenue_usd', 'cycles', 'final_soc_mwh',
            'vs_naive_usd', 'vs_perfect_usd', 'forecast_rmse', 'solver_status'.
        """
        bat = self._battery
        dt = 1.0  # hourly
        T = len(actual_prices)

        soc0 = initial_soc_mwh if initial_soc_mwh is not None else self._current_soc_mwh

        # Solve with forecasted prices → get schedule
        forecast_result = self._optimizer.solve(
            prices_mwh=forecast_prices,
            interval_hours=dt,
            initial_soc_mwh=soc0,
            constraints=None,
        )

        # Compute ACTUAL revenue by applying the forecasted schedule to actual prices
        deg_mwh = bat.degradation_cost_per_kwh * 1000.0
        actual_revenue = float(
            sum(
                actual_prices[t] * (forecast_result.discharge_mw[t] - forecast_result.charge_mw[t]) * dt
                - deg_mwh * (forecast_result.charge_mw[t] + forecast_result.discharge_mw[t]) * dt
                for t in range(T)
            )
        )

        # Naive baseline revenue at actual prices
        naive_result = self._optimizer.solve_baseline_naive(actual_prices, dt)

        # Perfect hindsight upper bound at actual prices
        perfect_result = self._optimizer.solve_perfect_hindsight(actual_prices, dt)

        # Advance SoC for next day
        self._current_soc_mwh = float(forecast_result.soc_mwh[-1])

        total_discharge_mwh = float(forecast_result.discharge_mw.sum() * dt)
        cycles = total_discharge_mwh / bat.capacity_mwh

        return {
            "date": date,
            "revenue_usd": actual_revenue,
            "cycles": cycles,
            "final_soc_mwh": float(forecast_result.soc_mwh[-1]),
            "vs_naive_usd": actual_revenue - naive_result.net_revenue_usd,
            "vs_perfect_usd": actual_revenue - perfect_result.net_revenue_usd,
            "forecast_rmse": float(np.sqrt(np.mean((forecast_prices - actual_prices) ** 2))),
            "solver_status": forecast_result.status,
            "naive_revenue_usd": naive_result.net_revenue_usd,
            "perfect_revenue_usd": perfect_result.net_revenue_usd,
        }

    def get_langchain_tools(self) -> list[Any]:
        """Return LangChain Tool definitions for the orchestrator.

        Returns:
            List of Tool objects: 'run_dispatch_optimization', 'get_current_soc'.
        """
        import json as _json
        from langchain_core.tools import tool

        agent_ref = self

        @tool
        def get_current_soc() -> str:
            """Return the current battery state-of-charge in MWh as JSON."""
            return _json.dumps({
                "current_soc_mwh": agent_ref._current_soc_mwh,
                "capacity_mwh": agent_ref._battery.capacity_mwh,
                "soc_pct": agent_ref._current_soc_mwh / agent_ref._battery.capacity_mwh,
            })

        @tool
        def get_battery_config() -> str:
            """Return battery physical parameters as JSON."""
            bat = agent_ref._battery
            return _json.dumps({
                "capacity_mwh": bat.capacity_mwh,
                "max_charge_mw": bat.max_charge_mw,
                "max_discharge_mw": bat.max_discharge_mw,
                "round_trip_efficiency": bat.round_trip_efficiency,
                "soc_min_mwh": bat.soc_min_mwh,
                "soc_max_mwh": bat.soc_max_mwh,
            })

        return [get_current_soc, get_battery_config]
