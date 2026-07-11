# Thesis Summary

The project originates from the MSc thesis **"Demand Estimation and Stock Management with Sales Information"** by Panagiotis Skartsilas.

The thesis studies inventory control for a single retailer selling one product over repeated periods. In each period the retailer chooses an order quantity before demand is known. Ordering too little creates lost sales and goodwill costs, while ordering too much creates leftover inventory and overage costs.

The mathematical foundation is the classical newsvendor model. The optimal order quantity is determined by the critical ratio:

```text
R = Cu / (Cu + Co)
```

For normally distributed demand, the benchmark quantity is:

```text
Q* = mu + sigma * Phi^-1(R)
```

The thesis compares three policies:

- **Uncensored newsvendor policy**: assumes exact demand is observed every period.
- **Burnetas-Smith policy**: adapts using a censored/uncensored indicator from the previous period.
- **Kaplan-Meier policy**: estimates an empirical distribution from censored sales observations.

The numerical experiments use Monte Carlo simulation to compare convergence of order quantities and realized profits against the theoretical newsvendor benchmark.

Main conclusions:

- The uncensored policy is strongest when complete demand data is available.
- Burnetas-Smith is simple and often converges quickly under censored demand.
- Kaplan-Meier can be slower early on because it needs enough historical observations.
- Cost ratios affect which censored-demand policy behaves better.

This repository refactors the thesis implementation into a reusable Python package and prepares it for further real-world inventory optimization scenarios.

