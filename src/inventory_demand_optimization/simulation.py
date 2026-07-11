"""Monte Carlo simulation engine for inventory ordering policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SimulationParameters
from .policies import (
    KaplanMeierState,
    burnetas_smith_quantity,
    newsvendor_quantity_from_history,
)
from .profit import expected_normal_newsvendor_profit, realized_profit


RESULT_COLUMNS = [
    "uncensored_quantity",
    "uncensored_profit",
    "burnetas_smith_quantity",
    "burnetas_smith_profit",
    "kaplan_meier_quantity",
    "kaplan_meier_profit",
]


@dataclass(frozen=True)
class SimulationResult:
    """Container for raw and averaged simulation output."""

    parameters: SimulationParameters
    raw_values: np.ndarray
    expected_profit: float

    @property
    def mean_values(self) -> np.ndarray:
        return self.raw_values.mean(axis=0)

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.mean_values, columns=RESULT_COLUMNS)
        frame.insert(0, "period", np.arange(len(frame)))
        frame["benchmark_quantity"] = self.parameters.newsvendor_quantity
        frame["benchmark_expected_profit"] = self.expected_profit

        for policy in ["uncensored", "burnetas_smith", "kaplan_meier"]:
            frame[f"{policy}_quantity_gap_pct"] = (
                100.0
                * abs(
                    frame[f"{policy}_quantity"] - self.parameters.newsvendor_quantity
                )
                / self.parameters.newsvendor_quantity
            )

        return frame

    def summary(self) -> pd.Series:
        frame = self.to_frame()
        last = frame.iloc[-1]
        return pd.Series(
            {
                "benchmark_quantity": self.parameters.newsvendor_quantity,
                "expected_profit": self.expected_profit,
                "final_uncensored_quantity": last["uncensored_quantity"],
                "final_burnetas_smith_quantity": last["burnetas_smith_quantity"],
                "final_kaplan_meier_quantity": last["kaplan_meier_quantity"],
                "final_uncensored_gap_pct": last["uncensored_quantity_gap_pct"],
                "final_burnetas_smith_gap_pct": last[
                    "burnetas_smith_quantity_gap_pct"
                ],
                "final_kaplan_meier_gap_pct": last[
                    "kaplan_meier_quantity_gap_pct"
                ],
            }
        )


def run_simulation(parameters: SimulationParameters | None = None) -> SimulationResult:
    """Run a Monte Carlo comparison of the three thesis ordering policies."""

    params = parameters or SimulationParameters()
    economics = params.economics
    rng = np.random.default_rng(params.seed)

    values = np.zeros((params.repetitions, params.periods, len(RESULT_COLUMNS)))
    benchmark_quantity = params.newsvendor_quantity

    for repetition in range(params.repetitions):
        demand = rng.normal(params.demand_mean, params.demand_std, params.periods)

        uncensored_quantity = benchmark_quantity
        burnetas_quantity = benchmark_quantity
        kaplan_quantity = benchmark_quantity
        kaplan_state = KaplanMeierState(initial_quantity=benchmark_quantity)

        for period in range(params.periods):
            values[repetition, period, 0] = newsvendor_quantity_from_history(
                demand[: period + 1],
                economics.critical_ratio,
            )
            values[repetition, period, 1] = realized_profit(
                demand[period],
                uncensored_quantity,
                economics,
            )
            uncensored_quantity = values[repetition, period, 0]

            values[repetition, period, 2] = burnetas_smith_quantity(
                demand[period],
                burnetas_quantity,
                economics.critical_ratio,
                period,
            )
            values[repetition, period, 3] = realized_profit(
                demand[period],
                burnetas_quantity,
                economics,
            )
            burnetas_quantity = values[repetition, period, 2]

            values[repetition, period, 4] = kaplan_state.update(
                demand[period],
                kaplan_quantity,
                economics,
                period,
            )
            values[repetition, period, 5] = realized_profit(
                demand[period],
                kaplan_quantity,
                economics,
            )
            kaplan_quantity = values[repetition, period, 4]

    expected_profit = expected_normal_newsvendor_profit(
        params.demand_mean,
        params.demand_std,
        economics,
    )
    return SimulationResult(params, values, expected_profit)

