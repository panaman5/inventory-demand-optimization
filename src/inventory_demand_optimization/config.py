"""Configuration objects for inventory demand simulations."""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import norm


@dataclass(frozen=True)
class EconomicParameters:
    """Economic inputs for the single-product inventory problem."""

    selling_price: float = 40.0
    unit_cost: float = 30.0
    salvage_value: float = 0.5
    goodwill_penalty: float = 1.0

    @property
    def underage_cost(self) -> float:
        return self.selling_price - self.unit_cost + self.goodwill_penalty

    @property
    def overage_cost(self) -> float:
        return self.unit_cost - self.salvage_value

    @property
    def critical_ratio(self) -> float:
        return self.underage_cost / (self.underage_cost + self.overage_cost)


@dataclass(frozen=True)
class SimulationParameters:
    """Simulation controls and demand-distribution assumptions."""

    economics: EconomicParameters = EconomicParameters()
    demand_mean: float = 200.0
    demand_std: float = 50.0
    periods: int = 500
    repetitions: int = 50
    seed: int = 2023

    @property
    def newsvendor_quantity(self) -> float:
        return self.demand_mean + self.demand_std * norm.ppf(
            self.economics.critical_ratio
        )

