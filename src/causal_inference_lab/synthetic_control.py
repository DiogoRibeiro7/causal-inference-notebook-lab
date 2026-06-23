"""Synthetic control utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class SyntheticControlResult:
    """Result object for a synthetic control fit."""

    estimated_effect: float
    pre_treatment_rmse: float
    weights: pd.DataFrame
    effects: pd.DataFrame
    treated_unit: str | int


def fit_synthetic_control(
    data: pd.DataFrame,
    treated_unit: int | str,
    unit_col: str = "unit",
    time_col: str = "time",
    outcome_col: str = "outcome",
    treatment_col: str = "treatment",
    pre_period_end: int | None = None,
) -> SyntheticControlResult:
    """Fit a synthetic control by minimizing pre-treatment squared error."""

    required = [unit_col, time_col, outcome_col, treatment_col]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    treated_mask = data[unit_col] == treated_unit
    if not treated_mask.any():
        raise ValueError("treated_unit is not present in data.")

    treated_data = data.loc[treated_mask]
    treated_times = np.sort(treated_data[time_col].unique())
    if pre_period_end is None:
        treated_after = treated_data.loc[treated_data[treatment_col] == 1, time_col].to_numpy()
        if treated_after.size == 0:
            raise ValueError("Cannot infer pre_period_end because treated unit has no treated periods.")
        pre_period_end = int(np.min(treated_after)) - 1
    if pre_period_end < int(np.min(treated_times)):
        raise ValueError("pre_period_end must be within observed time periods.")

    panel = data.pivot_table(index=unit_col, columns=time_col, values=outcome_col, aggfunc="mean")
    if panel.isnull().values.any():
        raise ValueError("Panel data must have complete outcome observations for all requested periods.")

    control_units = [unit for unit in panel.index if unit != treated_unit]
    if not control_units:
        raise ValueError("At least one control unit is required.")

    pre_periods = [t for t in treated_times if t <= pre_period_end]
    post_periods = [t for t in treated_times if t > pre_period_end]
    if len(pre_periods) == 0 or len(post_periods) == 0:
        raise ValueError("Need both pre- and post-treatment periods.")

    treated_pre = panel.loc[treated_unit, pre_periods].to_numpy(dtype=float)
    treated_post = panel.loc[treated_unit, post_periods].to_numpy(dtype=float)

    donor_pre = panel.loc[control_units, pre_periods].to_numpy(dtype=float)
    donor_post = panel.loc[control_units, post_periods].to_numpy(dtype=float)

    n_donors = donor_pre.shape[0]
    init = np.repeat(1.0 / n_donors, n_donors)

    def objective(weights: np.ndarray) -> float:
        pred = weights @ donor_pre
        return float(np.mean((treated_pre - pred) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bounds = [(0.0, 1.0) for _ in range(n_donors)]
    solution = minimize(objective, init, method="SLSQP", bounds=bounds, constraints=constraints)
    if not solution.success:
        raise ValueError(f"Synthetic control optimization failed: {solution.message}")

    weights = pd.DataFrame({"unit": control_units, "weight": solution.x})
    fitted_pre = weights["weight"].to_numpy() @ donor_pre
    pre_rmse = float(np.sqrt(np.mean((treated_pre - fitted_pre) ** 2)))

    fitted_post = weights["weight"].to_numpy() @ donor_post
    effects_series = treated_post - fitted_post
    estimated_effect = float(np.mean(effects_series))

    effects = pd.DataFrame(
        {
            "time": post_periods,
            "treated_outcome": treated_post,
            "synthetic_outcome": fitted_post,
            "effect": effects_series,
        }
    )

    return SyntheticControlResult(
        estimated_effect=estimated_effect,
        pre_treatment_rmse=pre_rmse,
        weights=weights,
        effects=effects,
        treated_unit=treated_unit,
    )
