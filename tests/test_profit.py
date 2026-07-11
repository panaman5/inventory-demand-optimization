from inventory_demand_optimization.config import EconomicParameters
from inventory_demand_optimization.profit import realized_profit


def test_realized_profit_when_demand_is_fully_satisfied():
    economics = EconomicParameters(
        selling_price=40,
        unit_cost=30,
        salvage_value=0.5,
        goodwill_penalty=1,
    )

    expected = -100 * 30 + 40 * 80 + 0.5 * 20

    assert realized_profit(demand=80, quantity=100, economics=economics) == expected


def test_realized_profit_when_stockout_occurs():
    economics = EconomicParameters(
        selling_price=40,
        unit_cost=30,
        salvage_value=0.5,
        goodwill_penalty=1,
    )

    expected = -100 * 30 + 40 * 100 - 1 * 20

    assert realized_profit(demand=120, quantity=100, economics=economics) == expected
