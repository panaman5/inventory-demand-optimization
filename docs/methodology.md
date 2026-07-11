# Research Methodology

This document explains the modeling choices behind the package implementation.
It is intended to make the repository understandable as an operations research
project, not only as Python code.

## Inventory Setting

The project studies a single-product, repeated-period inventory problem. At the
start of each period, a retailer chooses an order quantity before demand is
known. Demand is then realized during the period.

The decision balances two economic risks:

- ordering too few units creates lost sales and a goodwill penalty;
- ordering too many units creates leftover inventory and overage cost.

The classical newsvendor model gives the benchmark quantity for this tradeoff.

## Economic Parameters

The model uses four direct economic inputs:

- `selling_price`: revenue per sold unit;
- `unit_cost`: purchase cost per ordered unit;
- `salvage_value`: value recovered from an unsold unit;
- `goodwill_penalty`: penalty per unit of unmet demand.

These imply the newsvendor costs:

```text
underage_cost = selling_price - unit_cost + goodwill_penalty
overage_cost = unit_cost - salvage_value
critical_ratio = underage_cost / (underage_cost + overage_cost)
```

Both underage and overage costs must be positive for the optimization problem to
be meaningful.

## Demand Assumption

The simulation assumes independent normal demand across periods. This is not
because all real retail demand is normal, but because the normal assumption gives
a clear closed-form benchmark:

```text
Q* = mean + standard_deviation * inverse_normal_cdf(critical_ratio)
```

The adaptive policies are then compared against that benchmark.

## Observation Model

The important research feature is demand censoring.

If demand is below inventory, true demand is observed. If demand exceeds
inventory, the retailer only observes that all inventory sold. The true demand is
some unknown value above the order quantity.

In code:

```text
observed_sales = min(demand, order_quantity)
is_uncensored = demand <= order_quantity
```

This is why censored-demand policies are needed. They must learn from sales
records that sometimes hide true demand.

## Policies

### Uncensored Newsvendor Policy

This policy assumes full demand observations. It estimates the historical mean
and standard deviation from all observed demand values and applies the normal
newsvendor formula.

This is an information-rich baseline, not a realistic censored-sales policy.

### Burnetas-Smith Adaptive Policy

The Burnetas-Smith policy only uses whether the last period stocked out. If the
period stocked out, the next quantity increases. If demand was fully observed,
the next quantity decreases. The adjustment shrinks over time.

This makes the policy simple and distribution-free.

### Kaplan-Meier Censored-Demand Policy

The Kaplan-Meier policy stores censored sales observations and estimates a
complementary CDF using a product-limit style calculation. It then selects the
observed quantity whose estimated survival probability is closest to the
newsvendor target `1 - critical_ratio`.

The implementation also includes a support-expansion heuristic from the thesis
logic: when the selected maximum observed value is repeatedly censored, the
policy treats the observed support as too low and temporarily doubles the largest
observed sales value.

## Simulation Design

Each Monte Carlo repetition generates an independent demand path. All policies
face the same demand path within a repetition so their results are comparable.
The output is averaged across repetitions.

The simulation records:

- order quantity by policy;
- realized profit by policy;
- benchmark quantity;
- theoretical expected benchmark profit;
- percentage quantity gap from the benchmark.

## Current Research Limitations

The package is now structured for research use, but the following items should
be treated as future work before making strong empirical claims:

- reproduce every thesis figure with fixed scenario files;
- compare the refactored Kaplan-Meier logic against the original thesis script;
- validate the Kaplan-Meier estimator on small hand-calculated examples;
- add non-normal demand distributions for robustness;
- use real sales data where stockouts create actual censoring;
- report confidence intervals across Monte Carlo repetitions.

## Practical Interpretation

For a real inventory project, this repository is a starting point for answering:

- how much inventory should be ordered under uncertain demand?
- how much does censoring from stockouts distort demand learning?
- when is a simple adaptive policy enough?
- when does a history-based censored-data estimator add value?

