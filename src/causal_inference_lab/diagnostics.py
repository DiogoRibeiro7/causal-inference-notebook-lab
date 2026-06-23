"""Diagnostics for causal inference workflows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from causal_inference_lab.estimators import estimate_propensity_scores


def _validate_data_frame(data: object) -> None:
    """Validate that input is a non-empty DataFrame."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("data must not be empty.")


def _validate_covariates(data: pd.DataFrame, covariates: Sequence[str]) -> list[str]:
    """Validate covariate names and return a normalized list."""

    if isinstance(covariates, str):
        raise TypeError("covariates must be a sequence of column names, not a string.")

    try:
        covariate_list = list(covariates)
    except TypeError as exc:
        raise TypeError("covariates must be a sequence of column names.") from exc

    if not covariate_list:
        raise ValueError("covariates must not be empty.")
    if len(set(covariate_list)) != len(covariate_list):
        raise ValueError("covariates must be unique.")
    if not all(isinstance(name, str) for name in covariate_list):
        raise TypeError("covariates must be a sequence of column names.")

    missing = [column for column in covariate_list if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return covariate_list


def _validate_treatment_column(data: pd.DataFrame, treatment_col: str) -> np.ndarray:
    """Validate binary treatment values and return a boolean treated mask."""

    if not isinstance(treatment_col, str):
        raise TypeError("treatment_col must be a string.")
    if treatment_col not in data.columns:
        raise ValueError(f"Unknown treatment column: {treatment_col}")

    treatment = data[treatment_col].to_numpy(dtype=float)
    if not np.all(np.isfinite(treatment)):
        raise ValueError("treatment must be numeric and finite.")
    if np.any(np.isin(treatment, [0, 1], invert=True)):
        raise ValueError("treatment must be binary and encoded as 0/1.")
    if not np.any(treatment == 1) or not np.any(treatment == 0):
        raise ValueError("Both treated and control groups must contain observations.")

    return treatment == 1


def _validate_weights(weights: np.ndarray, expected_len: int) -> np.ndarray:
    """Validate non-negative finite weights with positive total mass."""

    weights_array = np.asarray(weights, dtype=float)
    if weights_array.ndim != 1:
        raise ValueError("weights must be one-dimensional.")
    if len(weights_array) != expected_len:
        raise ValueError("weights must have the same length as data.")
    if not np.all(np.isfinite(weights_array)):
        raise ValueError("weights must be finite.")
    if np.any(weights_array < 0):
        raise ValueError("weights must be non-negative.")
    if np.sum(weights_array) <= 0:
        raise ValueError("weights must have a positive sum.")
    return weights_array


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

    _validate_data_frame(data)
    if not isinstance(covariate, str):
        raise TypeError("covariate must be a string.")
    treated_mask = _validate_treatment_column(data, treatment_col=treatment_col)

    if covariate not in data.columns:
        raise ValueError(f"Unknown covariate: {covariate}")
    control_mask = ~treated_mask

    x = data[covariate].to_numpy(dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("covariate values must be finite.")

    if weights is None:
        treated_mean = float(np.mean(x[treated_mask]))
        control_mean = float(np.mean(x[control_mask]))
        treated_var = float(np.var(x[treated_mask]))
        control_var = float(np.var(x[control_mask]))
    else:
        weights = _validate_weights(weights, expected_len=len(data))
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

    _validate_data_frame(data)
    covariate_list = _validate_covariates(data, covariates)
    if not isinstance(treatment_col, str):
        raise TypeError("treatment_col must be a string.")

    rows = []
    for covariate in covariate_list:
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

    _validate_data_frame(data)
    covariate_list = _validate_covariates(data, covariates)
    treated_mask = _validate_treatment_column(data, treatment_col=treatment_col)
    treatment = treated_mask.astype(float)
    propensity = estimate_propensity_scores(data, covariate_list, treatment_col=treatment_col, clip=clip)
    return treatment / propensity + (1.0 - treatment) / (1.0 - propensity)


def overlap_summary(propensity_scores: np.ndarray) -> dict[str, float]:
    """Summarize propensity score overlap."""

    scores = np.asarray(propensity_scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("propensity_scores must be one-dimensional.")
    if scores.size == 0:
        raise ValueError("propensity_scores must not be empty.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("propensity_scores must be finite.")
    return {
        "min": float(np.min(scores)),
        "p01": float(np.quantile(scores, 0.01)),
        "p05": float(np.quantile(scores, 0.05)),
        "median": float(np.median(scores)),
        "p95": float(np.quantile(scores, 0.95)),
        "p99": float(np.quantile(scores, 0.99)),
        "max": float(np.max(scores)),
    }
