# Inventory Demand Optimization

Inventory demand optimization using newsvendor models, censored-demand estimation, and Monte Carlo simulation.

This project began as an MSc thesis in Business Mathematics at Athens University of Economics and Business. It studies a single-product inventory problem where a retailer must decide how much stock to order before uncertain demand is observed.

The repository turns the thesis work into a clean, reproducible Python project for data science and operations research portfolio use.

## What This Project Shows

- Newsvendor inventory optimization under uncertain demand
- Critical-ratio decision rules for balancing overage and underage costs
- Adaptive inventory policies when demand is censored by stockouts
- Kaplan-Meier style estimation for censored sales observations
- Monte Carlo simulation for policy comparison
- Profit and quantity benchmarking against the normal-demand optimum

## Methods Compared

The project compares three ordering policies:

- **Uncensored newsvendor policy**: estimates demand mean and standard deviation from fully observed demand history.
- **Burnetas-Smith adaptive policy**: updates the next order quantity using only whether the last period stocked out.
- **Kaplan-Meier policy**: builds an empirical survival estimate from censored sales data and selects the newsvendor quantile.

The benchmark is the classical normal-demand newsvendor quantity:

```text
Q* = mu + sigma * Phi^-1(Cu / (Cu + Co))
```

## Repository Structure

```text
.
|-- src/inventory_demand_optimization/
|   |-- config.py          # simulation and economic parameters
|   |-- policies.py        # ordering policies
|   |-- profit.py          # realized and expected profit functions
|   |-- simulation.py      # Monte Carlo simulation engine
|   `-- plotting.py        # reusable plotting helpers
|-- examples/
|   `-- run_default_simulation.py
|-- notebooks/
|   `-- project_walkthrough.ipynb
|-- docs/
|   |-- methodology.md
|   `-- thesis_summary.md
|-- tests/
|   |-- test_policies.py
|   `-- test_profit.py
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

The original thesis scripts are preserved separately during the cleanup process. The `src/` package is the public-facing implementation.

## Quick Start

Create a virtual environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the default simulation:

```bash
python examples/run_default_simulation.py
```

This creates:

- `outputs/default_simulation_results.csv`
- `outputs/default_policy_comparison.png`

## Minimal Python Usage

```python
from inventory_demand_optimization import SimulationParameters, run_simulation

params = SimulationParameters()
result = run_simulation(params)

print(result.summary())
```

## Thesis Origin

The underlying thesis is **"Demand Estimation and Stock Management with Sales Information"** by Panagiotis Skartsilas, submitted for the MSc in Business Mathematics at Athens University of Economics and Business in December 2022.

The original academic work focused on comparing newsvendor policies under fully observed and censored demand. This repository refactors that work into a reusable project structure suitable for further experiments and real-life inventory scenarios.

For modeling assumptions and policy details, see [docs/methodology.md](docs/methodology.md).

## Roadmap

- Reproduce all thesis numerical experiments with fixed random seeds
- Add scenario configuration files
- Add richer convergence diagnostics
- Validate the Kaplan-Meier policy against reference examples
- Extend the project to real demand datasets
- Add business-facing dashboards and operational case studies
