"""Ordering policies for the newsvendor inventory problem."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from .config import EconomicParameters


def newsvendor_quantity_from_history(
    demand_history: np.ndarray,
    critical_ratio: float,
) -> float:
    """Estimate a normal-demand newsvendor quantity from observed demand history."""

    history = np.asarray(demand_history, dtype=float)
    if history.size == 0:
        raise ValueError("demand_history must contain at least one observation")

    return float(history.mean() + history.std() * norm.ppf(critical_ratio))


def burnetas_smith_quantity(
    demand: float,
    previous_quantity: float,
    critical_ratio: float,
    period_index: int,
) -> float:
    """Burnetas-Smith adaptive update for censored-demand inventory control."""

    observed_exact_demand = demand <= previous_quantity
    indicator = 1.0 if observed_exact_demand else 0.0
    step = period_index + 1
    return float(previous_quantity * (1.0 - (indicator - critical_ratio) / step))


@dataclass
class KaplanMeierState:
    """State for a Kaplan-Meier style censored-demand ordering policy."""

    initial_quantity: float
    warmup_periods: int = 20
    observations: list[tuple[float, int]] = field(default_factory=list)
    selected_max_count: int = 0
    selected_max_censored_count: int = 0

    def update(
        self,
        demand: float,
        previous_quantity: float,
        economics: EconomicParameters,
        period_index: int,
    ) -> float:
        """Update the state with one demand observation and return next quantity."""

        observed_sale = min(float(demand), float(previous_quantity))
        uncensored = int(demand <= previous_quantity)
        self.observations.append((observed_sale, uncensored))

        if period_index <= self.warmup_periods:
            return float(self.initial_quantity)

        quantity = self._quantile_quantity(economics.critical_ratio)
        largest_observed = max(value for value, _ in self.observations)

        if np.isclose(quantity, largest_observed):
            self.selected_max_count += 1
            if uncensored == 0:
                self.selected_max_censored_count += 1

        threshold = economics.overage_cost / (
            2.0 * (economics.underage_cost + economics.overage_cost)
        )
        if (
            self.selected_max_count > 0
            and self.selected_max_censored_count / self.selected_max_count > threshold
        ):
            self.selected_max_count = 0
            self.selected_max_censored_count = 0
            return float(2.0 * largest_observed)

        return float(quantity)

    def _quantile_quantity(self, critical_ratio: float) -> float:
        """Choose the observed quantity closest to the KM complementary CDF target."""

        ordered = sorted(self.observations, key=lambda item: (item[0], -item[1]))
        n_obs = len(ordered)
        survival = 1.0
        target_survival = 1.0 - critical_ratio
        candidates: list[tuple[float, float]] = []

        for index, (value, uncensored) in enumerate(ordered):
            at_risk = n_obs - index
            if uncensored and at_risk > 0:
                survival *= (at_risk - 1.0) / at_risk
            candidates.append((value, survival))

        value, _ = min(
            candidates,
            key=lambda item: abs(item[1] - target_survival),
        )
        return float(value)

