"""Run the default inventory demand optimization simulation."""

from pathlib import Path

from inventory_demand_optimization import SimulationParameters, run_simulation
from inventory_demand_optimization.plotting import plot_policy_comparison


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    result = run_simulation(SimulationParameters())
    frame = result.to_frame()

    frame.to_csv(output_dir / "default_simulation_results.csv", index=False)
    fig, _ = plot_policy_comparison(result)
    fig.savefig(output_dir / "default_policy_comparison.png", dpi=150)

    print(result.summary().to_string())
    print(f"\nSaved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

