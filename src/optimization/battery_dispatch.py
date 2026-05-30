"""
Battery energy storage dispatch optimizer using Pyomo + HiGHS.

Solves a linear program (LP) over a rolling horizon (default 24 h) to find the
charge/discharge schedule that maximizes net revenue given:
    - Forecasted LMP prices (from PriceForecasterAgent)
    - Battery physical constraints (from BatteryConfig)
    - External constraints injected by RiskMonitor / MarketIntelAgent

Problem formulation (continuous LP):
    maximize  Σ_t [ p_t * (d_t - c_t) * Δt - deg_mwh * (c_t + d_t) * Δt ]
    subject to:
        SoC_t = SoC_{t-1} + eff_c * c_t * Δt - (1/eff_d) * d_t * Δt
        SoC_min ≤ SoC_t ≤ SoC_max
        0 ≤ c_t ≤ P_charge_max
        0 ≤ d_t ≤ P_discharge_max
        Σ_t d_t * Δt ≤ max_cycles * capacity  (cycle limit)
        external constraints (e.g., c_t = 0 during curtailment event)

Solver: HiGHS via `highspy` (open-source, no license required).
Fallback: GLPK if HiGHS is unavailable.
"""

from __future__ import annotations

import time
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
        return pd.DataFrame(
            {
                "charge_mw": self.charge_mw,
                "discharge_mw": self.discharge_mw,
                "soc_mwh": self.soc_mwh,
                "net_position_mw": self.discharge_mw - self.charge_mw,
            },
            index=times,
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
        self._battery = battery

        for candidate in ("appsi_highs", "glpk", "cbc"):
            if self._check_solver_available(candidate):
                self._solver_name = candidate
                break
        else:
            raise RuntimeError(
                "No LP solver found. Install highspy (`pip install highspy`) or glpk."
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
        import pyomo.environ as pyo

        t0 = time.time()
        bat = self._battery
        if initial_soc_mwh is None:
            initial_soc_mwh = bat.initial_soc_pct * bat.capacity_mwh
        if constraints is None:
            constraints = []

        T = len(prices_mwh)
        dt = interval_hours
        eff_c = bat.charge_efficiency
        eff_d = bat.discharge_efficiency
        deg_mwh = bat.degradation_cost_per_kwh * 1000.0  # $/kWh → $/MWh

        # ------------------------------------------------------------------ #
        # Build Pyomo model
        # ------------------------------------------------------------------ #
        m = pyo.ConcreteModel()
        m.T = pyo.Set(initialize=range(T))

        m.c = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, bat.max_charge_mw))
        m.d = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, bat.max_discharge_mw))
        m.soc = pyo.Var(m.T, domain=pyo.NonNegativeReals,
                        bounds=(bat.soc_min_mwh, bat.soc_max_mwh))

        # Objective
        p = prices_mwh  # local alias for closure

        def _obj(model: Any) -> Any:
            return sum(
                p[t] * (model.d[t] - model.c[t]) * dt
                - deg_mwh * (model.c[t] + model.d[t]) * dt
                for t in model.T
            )

        m.obj = pyo.Objective(rule=_obj, sense=pyo.maximize)

        # SoC dynamics
        soc0 = initial_soc_mwh

        def _soc_dyn(model: Any, t: int) -> Any:
            prev = soc0 if t == 0 else model.soc[t - 1]
            return model.soc[t] == prev + eff_c * model.c[t] * dt - (1.0 / eff_d) * model.d[t] * dt

        m.soc_dyn = pyo.Constraint(m.T, rule=_soc_dyn)

        # Daily cycle limit on total discharge energy
        m.cycle_lim = pyo.Constraint(
            expr=sum(m.d[t] * dt for t in m.T) <= bat.max_cycles_per_day * bat.capacity_mwh
        )

        # External constraints from other agents
        for i, con in enumerate(constraints):
            t_idx = con.interval_idx
            if not (0 <= t_idx < T):
                continue
            var = {"charge": m.c, "discharge": m.d, "soc": m.soc}.get(con.variable)
            if var is None:
                continue
            rhs = con.value_mw
            if con.operator == "eq":
                expr = var[t_idx] == rhs
            elif con.operator == "le":
                expr = var[t_idx] <= rhs
            elif con.operator == "ge":
                expr = var[t_idx] >= rhs
            else:
                continue
            m.add_component(f"ext_{i}", pyo.Constraint(expr=expr))

        # ------------------------------------------------------------------ #
        # Solve
        # ------------------------------------------------------------------ #
        solver = pyo.SolverFactory(self._solver_name)
        results = solver.solve(m, tee=False)

        try:
            status = str(results.termination_condition)
        except AttributeError:
            try:
                status = str(results.solver.termination_condition)
            except AttributeError:
                status = "unknown"

        is_ok = any(s in status.lower() for s in ("optimal", "feasible"))
        if not is_ok:
            return DispatchResult(
                charge_mw=np.zeros(T),
                discharge_mw=np.zeros(T),
                soc_mwh=np.full(T, initial_soc_mwh),
                net_revenue_usd=0.0,
                solve_time_s=time.time() - t0,
                status=status,
                active_constraints=constraints,
            )

        c_vals = np.array([pyo.value(m.c[t]) for t in range(T)], dtype=float)
        d_vals = np.array([pyo.value(m.d[t]) for t in range(T)], dtype=float)
        soc_vals = np.array([pyo.value(m.soc[t]) for t in range(T)], dtype=float)

        net_revenue = float(
            sum(
                p[t] * (d_vals[t] - c_vals[t]) * dt - deg_mwh * (c_vals[t] + d_vals[t]) * dt
                for t in range(T)
            )
        )

        return DispatchResult(
            charge_mw=c_vals,
            discharge_mw=d_vals,
            soc_mwh=soc_vals,
            net_revenue_usd=net_revenue,
            solve_time_s=time.time() - t0,
            status=status,
            active_constraints=constraints,
        )

    def solve_baseline_naive(
        self,
        prices_mwh: np.ndarray,
        interval_hours: float,
    ) -> DispatchResult:
        """Naive baseline: charge during hours 22–6, discharge during 14–20.

        Args:
            prices_mwh: LMP array — used only for P&L calculation, not scheduling.
            interval_hours: Interval duration in hours.

        Returns:
            DispatchResult with the hardcoded schedule and resulting revenue.
        """
        t0 = time.time()
        T = len(prices_mwh)
        bat = self._battery
        dt = interval_hours
        intervals_per_hour = max(1, round(1.0 / dt))

        c = np.zeros(T)
        d = np.zeros(T)

        for t in range(T):
            hour = (t // intervals_per_hour) % 24
            if hour >= 22 or hour < 6:
                c[t] = bat.max_charge_mw
            elif 14 <= hour < 20:
                d[t] = bat.max_discharge_mw

        # Enforce SoC bounds
        soc = np.zeros(T)
        current_soc = bat.initial_soc_pct * bat.capacity_mwh
        eff_c = bat.charge_efficiency
        eff_d = bat.discharge_efficiency

        for t in range(T):
            headroom = (bat.soc_max_mwh - current_soc) / (eff_c * dt) if eff_c * dt > 0 else 0.0
            slack = (current_soc - bat.soc_min_mwh) * eff_d / dt if dt > 0 else 0.0
            actual_c = min(c[t], max(headroom, 0.0))
            actual_d = min(d[t], max(slack, 0.0))
            current_soc = current_soc + eff_c * actual_c * dt - (1.0 / eff_d) * actual_d * dt
            soc[t] = current_soc
            c[t] = actual_c
            d[t] = actual_d

        deg_mwh = bat.degradation_cost_per_kwh * 1000.0
        net_revenue = float(
            sum(
                prices_mwh[t] * (d[t] - c[t]) * dt - deg_mwh * (c[t] + d[t]) * dt
                for t in range(T)
            )
        )

        return DispatchResult(
            charge_mw=c,
            discharge_mw=d,
            soc_mwh=soc,
            net_revenue_usd=net_revenue,
            solve_time_s=time.time() - t0,
            status="naive_heuristic",
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
        return self.solve(
            prices_mwh=actual_prices_mwh,
            interval_hours=interval_hours,
            initial_soc_mwh=self._battery.initial_soc_pct * self._battery.capacity_mwh,
            constraints=None,
        )

    @staticmethod
    def _check_solver_available(solver_name: str) -> bool:
        """Return True if the given Pyomo solver is installed and callable.

        Args:
            solver_name: Pyomo solver string (e.g., 'appsi_highs', 'glpk').

        Returns:
            True if available, False otherwise.
        """
        try:
            import pyomo.environ as pyo
            return bool(pyo.SolverFactory(solver_name).available())
        except Exception:
            return False
