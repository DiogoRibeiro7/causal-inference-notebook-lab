"""Difference-in-differences estimators and simple diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from causal_inference_lab.estimators import EffectEstimate


@dataclass(frozen=True)
class DifferenceInDifferencesResult:
    """Container for a panel-based DiD estimate and diagnostics."""

    effect: EffectEstimate
    pre_treated_mean: float
    post_treated_mean: float
    pre_control_mean: float
    post_control_mean: float
    pre_trend_slope_difference: float
    pre_trend_p_value: float


def _validate_did_inputs(
    data: pd.DataFrame,
    group_col: str,
    post_col: str,
    time_col: str,
    outcome_col: str,
) -> None:
    required = [group_col, post_col, time_col, outcome_col]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if data.empty:
        raise ValueError("data must not be empty.")

    if data[[group_col, post_col]].isnull().any().any():
        raise ValueError("group_col and post_col must not contain missing values.")

    if data[time_col].nunique() < 2:
        raise ValueError("Need at least two unique time periods.")

    if not set(data[post_col].dropna().unique()).issubset({0, 1}):
        raise ValueError("post_col must be binary 0/1.")

    if not data[post_col].isin([0, 1]).all():
        raise ValueError("post_col must be binary 0/1.")

    group_values = set(data[group_col].dropna().unique())
    if len(group_values) != 2:
        raise ValueError(f"group_col must have exactly 2 groups; observed: {sorted(group_values)}.")

    if not set(data[group_col].dropna().unique()).issubset({0, 1}):
        raise ValueError("group_col must be binary and encoded as 0/1.")

    if data[group_col].nunique() < 2:
        raise ValueError("Both treated and control groups are required.")

    if data[post_col].sum() == 0 or (1 - data[post_col]).sum() == 0:
        raise ValueError("Both pre and post periods are required.")

    treatment_like = set(data[group_col].dropna().unique())
    if treatment_like != {0, 1}:
        raise ValueError("group_col must be encoded as 0/1.")


def _pre_trend_diagnostics(
    data: pd.DataFrame,
    group_col: str,
    post_col: str,
    time_col: str,
    outcome_col: str,
) -> tuple[float, float]:
    pre_data = data[data[post_col] == 0]
    if pre_data[time_col].nunique() < 2:
        return 0.0, 1.0

    design = pd.DataFrame(
        {
            "time": pre_data[time_col].astype(float),
            "group": pre_data[group_col].astype(float),
        }
    )
    design["interaction"] = design["time"] * design["group"]
    model = sm.OLS(
        pre_data[outcome_col].astype(float),
        sm.add_constant(design, has_constant="add"),
    ).fit()

    slope_diff = float(model.params["interaction"])
    p_value = float(model.pvalues["interaction"])
    return slope_diff, p_value


def difference_in_differences(
    data: pd.DataFrame,
    group_col: str = "treated_group",
    time_col: str = "time",
    post_col: str = "post",
    outcome_col: str = "outcome",
) -> DifferenceInDifferencesResult:
    """Estimate the average treatment effect with a two-way DiD contrast.

    The estimator assumes:
    - a binary treated/control indicator in ``group_col``,
    - a binary pre/post indicator in ``post_col``,
    - panel balance is not required, but at least one pre and one post observation
      is required for both groups.
    """

    _validate_did_inputs(
        data,
        group_col=group_col,
        post_col=post_col,
        time_col=time_col,
        outcome_col=outcome_col,
    )

    treated_mask = data[group_col] == 1
    control_mask = data[group_col] == 0
    pre_mask = data[post_col] == 0
    post_mask = data[post_col] == 1

    pre_treated = data.loc[treated_mask & pre_mask, outcome_col].to_numpy(dtype=float)
    post_treated = data.loc[treated_mask & post_mask, outcome_col].to_numpy(dtype=float)
    pre_control = data.loc[control_mask & pre_mask, outcome_col].to_numpy(dtype=float)
    post_control = data.loc[control_mask & post_mask, outcome_col].to_numpy(dtype=float)

    if (
        pre_treated.size == 0
        or post_treated.size == 0
        or pre_control.size == 0
        or post_control.size == 0
    ):
        raise ValueError("Each of treated/control x pre/post groups must contain observations.")

    delta_treated = float(np.mean(post_treated) - np.mean(pre_treated))
    delta_control = float(np.mean(post_control) - np.mean(pre_control))
    did_estimate = float(delta_treated - delta_control)

    slope_diff, p_value = _pre_trend_diagnostics(
        data=data,
        group_col=group_col,
        post_col=post_col,
        time_col=time_col,
        outcome_col=outcome_col,
    )

    effect = EffectEstimate(
        estimate=did_estimate,
        estimator="difference_in_differences",
        estimand="ATT",
        n_observations=int(len(data)),
    )

    return DifferenceInDifferencesResult(
        effect=effect,
        pre_treated_mean=float(np.mean(pre_treated)),
        post_treated_mean=float(np.mean(post_treated)),
        pre_control_mean=float(np.mean(pre_control)),
        post_control_mean=float(np.mean(post_control)),
        pre_trend_slope_difference=slope_diff,
        pre_trend_p_value=p_value,
    )
