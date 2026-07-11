import numpy as np

from inventory_demand_optimization.policies import (
    burnetas_smith_quantity,
    newsvendor_quantity_from_history,
)


def test_newsvendor_quantity_from_constant_history_is_mean():
    demand = np.array([100, 100, 100])

    assert newsvendor_quantity_from_history(demand, critical_ratio=0.8) == 100


def test_burnetas_smith_increases_after_stockout():
    quantity = burnetas_smith_quantity(
        demand=120,
        previous_quantity=100,
        critical_ratio=0.4,
        period_index=0,
    )

    assert quantity > 100


def test_burnetas_smith_decreases_after_demand_is_observed():
    quantity = burnetas_smith_quantity(
        demand=80,
        previous_quantity=100,
        critical_ratio=0.4,
        period_index=0,
    )

    assert quantity < 100

