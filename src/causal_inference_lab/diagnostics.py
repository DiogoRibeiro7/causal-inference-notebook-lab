"""Diagnostics for causal inference workflows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from causal_inference_lab.estimators import estimate_propensity_scores


def _weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    """Return weighted variance with non-negative finite weights."""

    if len(values) == 0:
        return 0.0
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length.")
    if np.any(~np.isfinite(weights)) or np.any(~np.isfinite(values)):
        raise ValueError("values and weights must be finite.")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative.")
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        raise ValueError("weights must have a positive sum for each group.")

    mean = float(np.average(values, weights=weights))
    return float(np.average((values - mean) ** 2, weights=weights))


def standardized_mean_difference(
    data: pd.DataFrame,
    covariate: str,
    treatment_col: str = "treatment",
    weights: np.ndarray | None = None,
) -> float:
    """Compute standardized mean difference for one covariate.

    Args:
        data: Input data.
        covariate: Covariate to compare.
        treatment_col: Binary treatment column.
        weights: Optional observation weights.

    Returns:
        Standardized mean difference between treated and control groups.
    """

    if covariate not in data.columns:
        raise ValueError(f"Unknown covariate: {covariate}")
    if treatment_col not in data.columns:
        raise ValueError(f"Unknown treatment column: {treatment_col}")

    treated_mask = data[treatment_col].to_numpy() == 1
    control_mask = ~treated_mask
    if not treated_mask.any() or not control_mask.any():
        raise ValueError("Both treated and control groups must contain observations.")

    x = data[covariate].to_numpy(dtype=float)

    if weights is None:
        treated_mean = float(np.mean(x[treated_mask]))
        control_mean = float(np.mean(x[control_mask]))
        treated_var = float(np.var(x[treated_mask]))
        control_var = float(np.var(x[control_mask]))
    else:
        if len(weights) != len(data):
            raise ValueError("weights must have the same length as data.")
        weights = np.asarray(weights, dtype=float)
        treated_mean = float(np.average(x[treated_mask], weights=weights[treated_mask]))
        control_mean = float(np.average(x[control_mask], weights=weights[control_mask]))
        treated_var = _weighted_variance(x[treated_mask], weights[treated_mask])
        control_var = _weighted_variance(x[control_mask], weights[control_mask])

    pooled_sd = float(np.sqrt((treated_var + control_var) / 2.0))
    if pooled_sd == 0.0:
        return 0.0

    return (treated_mean - control_mean) / pooled_sd


def balance_table(
    data: pd.DataFrame,
    covariates: Sequence[str],
    treatment_col: str = "treatment",
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a balance table with standardized mean differences."""

    rows = []
    for covariate in covariates:
        rows.append(
            {
                "covariate": covariate,
                "smd": standardized_mean_difference(
                    data=data,
                    covariate=covariate,
                    treatment_col=treatment_col,
                    weights=weights,
                ),
            }
        )
    return pd.DataFrame(rows).assign(abs_smd=lambda frame: frame["smd"].abs())


def ipw_weights(
    data: pd.DataFrame,
    covariates: Sequence[str],
    treatment_col: str = "treatment",
    clip: float = 0.01,
) -> np.ndarray:
    """Return inverse probability weights for ATE estimation."""

    treatment = data[treatment_col].to_numpy(dtype=float)
    propensity = estimate_propensity_scores(data, covariates, treatment_col, clip=clip)
    return treatment / propensity + (1.0 - treatment) / (1.0 - propensity)


def overlap_summary(propensity_scores: np.ndarray) -> dict[str, float]:
    """Summarize propensity score overlap."""

    if propensity_scores.ndim != 1:
        raise ValueError("propensity_scores must be a one-dimensional array.")

    return {
        "min": float(np.min(propensity_scores)),
        "p01": float(np.quantile(propensity_scores, 0.01)),
        "p05": float(np.quantile(propensity_scores, 0.05)),
        "median": float(np.median(propensity_scores)),
        "p95": float(np.quantile(propensity_scores, 0.95)),
        "p99": float(np.quantile(propensity_scores, 0.99)),
        "max": float(np.max(propensity_scores)),
    }
