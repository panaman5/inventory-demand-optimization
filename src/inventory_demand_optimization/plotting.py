"""Plotting helpers for simulation results."""

from __future__ import annotations

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from .simulation import SimulationResult


def plot_policy_comparison(
    result: SimulationResult,
    smoothing_sigma: float = 20.0,
):
    """Create quantity, profit, and benchmark-gap plots for a simulation."""

    frame = result.to_frame()
    periods = frame["period"]

    def smooth(column: str):
        return gaussian_filter1d(frame[column].to_numpy(), sigma=smoothing_sigma)

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)

    axes[0].plot(periods, smooth("uncensored_quantity"), label="Uncensored")
    axes[0].plot(periods, smooth("burnetas_smith_quantity"), label="Burnetas-Smith")
    axes[0].plot(periods, smooth("kaplan_meier_quantity"), label="Kaplan-Meier")
    axes[0].axhline(
        result.parameters.newsvendor_quantity,
        color="black",
        linestyle="--",
        label="Newsvendor benchmark",
    )
    axes[0].set_title("Order Quantity by Policy")
    axes[0].set_xlabel("Period")
    axes[0].set_ylabel("Quantity")
    axes[0].legend()

    axes[1].plot(periods, smooth("uncensored_profit"), label="Uncensored")
    axes[1].plot(periods, smooth("burnetas_smith_profit"), label="Burnetas-Smith")
    axes[1].plot(periods, smooth("kaplan_meier_profit"), label="Kaplan-Meier")
    axes[1].axhline(
        result.expected_profit,
        color="black",
        linestyle="--",
        label="Expected profit benchmark",
    )
    axes[1].set_title("Realized Profit by Policy")
    axes[1].set_xlabel("Period")
    axes[1].set_ylabel("Profit")
    axes[1].legend()

    axes[2].plot(
        periods,
        smooth("uncensored_quantity_gap_pct"),
        label="Uncensored",
    )
    axes[2].plot(
        periods,
        smooth("burnetas_smith_quantity_gap_pct"),
        label="Burnetas-Smith",
    )
    axes[2].plot(
        periods,
        smooth("kaplan_meier_quantity_gap_pct"),
        label="Kaplan-Meier",
    )
    axes[2].set_title("Percentage Gap From Benchmark Quantity")
    axes[2].set_xlabel("Period")
    axes[2].set_ylabel("Gap (%)")
    axes[2].legend()

    return fig, axes

