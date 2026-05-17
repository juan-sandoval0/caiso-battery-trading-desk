"""
Battery energy storage system (BESS) physical and financial parameters.

All values represent the virtual battery used in simulation and optimization.
Change these constants to model different BESS configurations.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryConfig:
    """Physical and financial parameters for the virtual BESS.

    Attributes:
        capacity_mwh: Usable energy capacity in MWh.
        max_charge_mw: Maximum charge power in MW.
        max_discharge_mw: Maximum discharge power in MW.
        charge_efficiency: One-way charge efficiency (0–1).
        discharge_efficiency: One-way discharge efficiency (0–1).
        soc_min_pct: Minimum state-of-charge as fraction of capacity.
        soc_max_pct: Maximum state-of-charge as fraction of capacity.
        degradation_cost_per_kwh: Variable O&M cost per kWh cycled ($/kWh).
        max_cycles_per_day: Maximum full equivalent cycles per day.
        initial_soc_pct: Starting SoC fraction for each simulation episode.
    """

    capacity_mwh: float = 4.0
    max_charge_mw: float = 1.0
    max_discharge_mw: float = 1.0
    charge_efficiency: float = 0.935
    discharge_efficiency: float = 0.935
    soc_min_pct: float = 0.10
    soc_max_pct: float = 0.90
    degradation_cost_per_kwh: float = 0.05
    max_cycles_per_day: float = 1.0
    initial_soc_pct: float = 0.50

    @property
    def round_trip_efficiency(self) -> float:
        """Combined round-trip efficiency: charge_eff × discharge_eff."""
        return self.charge_efficiency * self.discharge_efficiency

    @property
    def soc_min_mwh(self) -> float:
        """Absolute minimum SoC in MWh."""
        return self.soc_min_pct * self.capacity_mwh

    @property
    def soc_max_mwh(self) -> float:
        """Absolute maximum SoC in MWh."""
        return self.soc_max_pct * self.capacity_mwh


# Module-level default — import this in other modules
DEFAULT_BATTERY = BatteryConfig()
