"""Ordering policies for the newsvendor inventory problem.

The module keeps the policy functions close to their mathematical definitions:

- the uncensored policy estimates normal-demand parameters from full history;
- Burnetas-Smith adapts using only a stockout indicator;
- Kaplan-Meier estimates a survival curve from censored sales observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from .config import EconomicParameters


def newsvendor_quantity_from_history(
    demand_history: np.ndarray,
    critical_ratio: float,
) -> float:
    """Estimate the normal-demand newsvendor quantity from full demand history.

    This is the full-information benchmark policy used in the simulation. It
    assumes the retailer observes actual demand even when demand exceeds the
    order quantity.
    """

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
    """Burnetas-Smith adaptive update for censored-demand inventory control.

    The update is:

    Q_next = Q_previous * (1 - (Y - R) / t)

    where Y=1 when demand is fully observed and Y=0 after a stockout. A stockout
    therefore increases the next quantity, while a fully observed period lowers
    it. The adjustment size shrinks with time.
    """

    observed_exact_demand = demand <= previous_quantity
    indicator = 1.0 if observed_exact_demand else 0.0
    step = period_index + 1
    return float(previous_quantity * (1.0 - (indicator - critical_ratio) / step))


@dataclass(frozen=True)
class CensoredDemandObservation:
    """One sales observation under inventory censoring."""

    observed_sales: float
    is_uncensored: bool


@dataclass
class KaplanMeierState:
    """State for a Kaplan-Meier style censored-demand ordering policy.

    The retailer observes sales rather than true demand:

    observed_sales = min(demand, previous_quantity)

    If demand exceeds inventory, the observation is right-censored because true
    demand could be any value above the observed sales. The state stores these
    observations, builds a product-limit survival estimate, and selects the
    quantity whose estimated survival probability is closest to 1 - R.
    """

    initial_quantity: float
    warmup_periods: int = 20
    observations: list[CensoredDemandObservation] = field(default_factory=list)
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

        observation = self._observe(demand, previous_quantity)
        self.observations.append(observation)

        if period_index <= self.warmup_periods:
            return float(self.initial_quantity)

        survival_curve = self.complementary_cdf_points()
        quantity = self._quantity_from_survival_curve(
            survival_curve,
            economics.critical_ratio,
        )

        if self._support_expansion_triggered(quantity, observation, economics):
            self.selected_max_count = 0
            self.selected_max_censored_count = 0
            return float(2.0 * self.largest_observed_sales)

        return float(quantity)

    @property
    def largest_observed_sales(self) -> float:
        """Largest sales value seen so far."""

        return max(observation.observed_sales for observation in self.observations)

    def complementary_cdf_points(self) -> list[tuple[float, float]]:
        """Return product-limit estimates of the complementary CDF.

        Each tuple is `(observed_sales, estimated_survival_probability)`. Ties are
        ordered with uncensored observations first, matching the convention used
        in the thesis algorithm.
        """

        ordered = sorted(
            self.observations,
            key=lambda item: (item.observed_sales, not item.is_uncensored),
        )
        n_obs = len(ordered)
        survival = 1.0
        candidates: list[tuple[float, float]] = []

        for index, observation in enumerate(ordered):
            at_risk = n_obs - index
            if observation.is_uncensored and at_risk > 0:
                survival *= (at_risk - 1.0) / at_risk
            candidates.append((observation.observed_sales, survival))

        return candidates

    @staticmethod
    def _observe(
        demand: float,
        previous_quantity: float,
    ) -> CensoredDemandObservation:
        return CensoredDemandObservation(
            observed_sales=min(float(demand), float(previous_quantity)),
            is_uncensored=demand <= previous_quantity,
        )

    @staticmethod
    def _quantity_from_survival_curve(
        survival_curve: list[tuple[float, float]],
        critical_ratio: float,
    ) -> float:
        """Choose the observed quantity closest to the newsvendor survival target."""

        target_survival = 1.0 - critical_ratio

        value, _ = min(
            survival_curve,
            key=lambda item: abs(item[1] - target_survival),
        )
        return float(value)

    def _support_expansion_triggered(
        self,
        selected_quantity: float,
        latest_observation: CensoredDemandObservation,
        economics: EconomicParameters,
    ) -> bool:
        """Heuristic from the thesis to respond when the observed support is too low."""

        if np.isclose(selected_quantity, self.largest_observed_sales):
            self.selected_max_count += 1
            if not latest_observation.is_uncensored:
                self.selected_max_censored_count += 1

        if self.selected_max_count == 0:
            return False

        censored_share_at_max = (
            self.selected_max_censored_count / self.selected_max_count
        )
        threshold = economics.overage_cost / (
            2.0 * (economics.underage_cost + economics.overage_cost)
        )
        return censored_share_at_max > threshold
