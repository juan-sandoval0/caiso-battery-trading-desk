"""
Battery energy storage dispatch optimizer using Pyomo + HiGHS.

Solves a linear program (LP) over a rolling horizon (default 24 h) to find the
charge/discharge schedule that maximizes net revenue given:
    - Forecasted LMP prices (from PriceForecasterAgent)
    - Battery physical constraints (from BatteryConfig)
    - External constraints injected by RiskMonitor / MarketIntelAgent

Problem formulation (continuous LP — no binary variables needed for 1-cycle limit):
    maximize  Σ_t [ p_t * d_t * eff_d - p_t * c_t / eff_c - deg * (c_t + d_t) ]
    subject to:
        SoC_t = SoC_{t-1} + c_t * eff_c * Δt - d_t / eff_d * Δt
        SoC_min ≤ SoC_t ≤ SoC_max
        0 ≤ c_t ≤ P_charge_max
        0 ≤ d_t ≤ P_discharge_max
        Σ_t d_t * Δt ≤ max_cycles * capacity  (cycle limit)
        external constraints (e.g., c_t = 0 during curtailment event)

Solver: HiGHS via `highspy` (open-source, no license required).
Fallback: GLPK if HiGHS is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.config.battery import BatteryConfig, DEFAULT_BATTERY


@dataclass
class DispatchConstraint:
    """An external constraint injected into the optimizer by another agent.

    Attributes:
        interval_idx: Time step index (0-based) this constraint applies to.
        variable: 'charge' | 'discharge' | 'soc'
        operator: 'eq' | 'le' | 'ge'
        value_mw: Right-hand-side value in MW (or MWh for soc).
        source: Which agent imposed this constraint (for logging/audit).
        reason: Human-readable rationale.
    """

    interval_idx: int
    variable: str
    operator: str
    value_mw: float
    source: str
    reason: str = ""


@dataclass
class DispatchResult:
    """Output of the LP solver for one planning horizon.

    Attributes:
        charge_mw: Charge power schedule (MW), indexed by interval.
        discharge_mw: Discharge power schedule (MW), indexed by interval.
        soc_mwh: State-of-charge trajectory (MWh), indexed by interval.
        net_revenue_usd: Total expected revenue over the horizon (USD).
        solve_time_s: Wall-clock solver time in seconds.
        status: Solver status string ('optimal', 'infeasible', etc.).
        active_constraints: List of DispatchConstraints applied in this solve.
    """

    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    soc_mwh: np.ndarray
    net_revenue_usd: float
    solve_time_s: float
    status: str
    active_constraints: list[DispatchConstraint] = field(default_factory=list)

    def to_dataframe(self, times: pd.DatetimeIndex) -> pd.DataFrame:
        """Return the schedule as a DataFrame indexed by time.

        Args:
            times: DatetimeIndex matching the length of charge_mw.

        Returns:
            DataFrame with columns [charge_mw, discharge_mw, soc_mwh, net_position_mw].
        """
        raise NotImplementedError(
            # Build DataFrame from self arrays, add net_position_mw = discharge - charge.
        )


class BatteryDispatchOptimizer:
    """Pyomo + HiGHS LP solver for battery charge/discharge scheduling.

    Usage::

        optimizer = BatteryDispatchOptimizer(battery=DEFAULT_BATTERY)
        result = optimizer.solve(
            prices_mwh=np.array([...]),    # $/MWh for each interval
            interval_hours=1.0,            # 1.0 for hourly, 1/12 for 5-min
            initial_soc_mwh=2.0,
            constraints=[...],             # injected by RiskMonitor
        )
    """

    def __init__(self, battery: BatteryConfig = DEFAULT_BATTERY) -> None:
        """Initialize the optimizer with a battery configuration.

        Args:
            battery: Physical and financial BESS parameters.
        """
        raise NotImplementedError(
            # Store battery as self._battery.
            # Detect solver availability: try HiGHS first, fall back to GLPK.
            # self._solver_name = 'appsi_highs' if highs available else 'glpk'
        )

    def solve(
        self,
        prices_mwh: np.ndarray,
        interval_hours: float,
        initial_soc_mwh: float | None = None,
        constraints: list[DispatchConstraint] | None = None,
    ) -> DispatchResult:
        """Solve the dispatch LP for one planning horizon.

        Args:
            prices_mwh: Forecasted LMP in $/MWh for each interval (length = T).
            interval_hours: Duration of each interval in hours (e.g., 1.0 or 1/12).
            initial_soc_mwh: Starting SoC in MWh. Defaults to 50% of capacity.
            constraints: External constraints from RiskMonitor / MarketIntelAgent.

        Returns:
            DispatchResult with the optimal schedule and solve metadata.
        """
        raise NotImplementedError(
            # 1. Build Pyomo ConcreteModel with Sets, Params, Vars, Objective, Constraints.
            # 2. Apply each DispatchConstraint in `constraints`.
            # 3. Invoke SolverFactory(self._solver_name).solve(model).
            # 4. Extract Var values into numpy arrays.
            # 5. Compute net_revenue_usd = sum(price * discharge - price * charge - deg).
            # 6. Return DispatchResult.
        )

    def solve_baseline_naive(
        self,
        prices_mwh: np.ndarray,
        interval_hours: float,
    ) -> DispatchResult:
        """Naive baseline: charge during hours 22–6, discharge during 14–20.

        Args:
            prices_mwh: LMP array (ignored for scheduling, used for P&L calc only).
            interval_hours: Interval duration in hours.

        Returns:
            DispatchResult with hardcoded schedule and resulting revenue.
        """
        raise NotImplementedError(
            # Build charge/discharge arrays based on hour-of-day masks.
            # Compute SoC trajectory respecting capacity bounds.
            # Compute revenue using prices_mwh.
        )

    def solve_perfect_hindsight(
        self,
        actual_prices_mwh: np.ndarray,
        interval_hours: float,
    ) -> DispatchResult:
        """Upper-bound baseline: solve LP with perfect price knowledge.

        Used to compute the maximum achievable revenue on any given day.

        Args:
            actual_prices_mwh: True realized LMP (not forecast).
            interval_hours: Interval duration in hours.

        Returns:
            DispatchResult representing the theoretical maximum revenue.
        """
        raise NotImplementedError(
            # Same as solve() but with actual prices — no forecasting error.
        )

    @staticmethod
    def _check_solver_available(solver_name: str) -> bool:
        """Return True if the given Pyomo solver is installed and callable.

        Args:
            solver_name: Pyomo solver string (e.g., 'appsi_highs', 'glpk').

        Returns:
            True if available, False otherwise.
        """
        raise NotImplementedError(
            # from pyomo.environ import SolverFactory
            # return SolverFactory(solver_name).available()
        )
