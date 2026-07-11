"""Configuration objects for inventory demand simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from scipy.stats import norm


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_positive(name: str, value: float) -> None:
    _require_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(name: str, value: float) -> None:
    _require_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class EconomicParameters:
    """Economic inputs for one selling period.

    The implied newsvendor costs must both be positive:

    - underage cost = selling_price - unit_cost + goodwill_penalty
    - overage cost = unit_cost - salvage_value
    """

    selling_price: float = 40.0
    unit_cost: float = 30.0
    salvage_value: float = 0.5
    goodwill_penalty: float = 1.0

    def __post_init__(self) -> None:
        _require_positive("selling_price", self.selling_price)
        _require_positive("unit_cost", self.unit_cost)
        _require_non_negative("salvage_value", self.salvage_value)
        _require_non_negative("goodwill_penalty", self.goodwill_penalty)
        _require_positive("underage_cost", self.underage_cost)
        _require_positive("overage_cost", self.overage_cost)

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
    """Simulation controls and demand-distribution assumptions.

    Demand is modeled as normal in order to compare adaptive policies against
    the closed-form newsvendor benchmark used in the thesis.
    """

    economics: EconomicParameters = field(default_factory=EconomicParameters)
    demand_mean: float = 200.0
    demand_std: float = 50.0
    periods: int = 500
    repetitions: int = 50
    seed: int = 2023

    def __post_init__(self) -> None:
        _require_positive("demand_mean", self.demand_mean)
        _require_positive("demand_std", self.demand_std)

        if self.periods <= 0:
            raise ValueError("periods must be positive")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")

    @property
    def newsvendor_quantity(self) -> float:
        return self.demand_mean + self.demand_std * norm.ppf(
            self.economics.critical_ratio
        )
