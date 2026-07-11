"""Profit functions for single-period inventory decisions."""

from __future__ import annotations

from scipy.stats import norm

from .config import EconomicParameters


def realized_profit(
    demand: float,
    quantity: float,
    economics: EconomicParameters,
) -> float:
    """Calculate one-period realized profit for a demand and order quantity.

    If demand is fully satisfied, leftover units receive salvage value. If the
    retailer stocks out, all ordered units are sold and unmet demand incurs a
    goodwill penalty.
    """

    if demand <= quantity:
        return (
            -quantity * economics.unit_cost
            + economics.selling_price * demand
            + economics.salvage_value * (quantity - demand)
        )

    return (
        -quantity * economics.unit_cost
        + economics.selling_price * quantity
        - economics.goodwill_penalty * (demand - quantity)
    )


def expected_normal_newsvendor_profit(
    demand_mean: float,
    demand_std: float,
    economics: EconomicParameters,
) -> float:
    """Expected optimal profit for normal demand under the newsvendor benchmark.

    At the optimal newsvendor quantile, the normal-loss expression simplifies to
    the mean margin less the critical standard-normal density term.
    """

    z_value = norm.ppf(economics.critical_ratio)
    return (
        (economics.selling_price - economics.unit_cost) * demand_mean
        - (
            economics.selling_price
            - economics.salvage_value
            + economics.goodwill_penalty
        )
        * norm.pdf(z_value)
        * demand_std
    )
