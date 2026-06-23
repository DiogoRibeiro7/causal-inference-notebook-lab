"""Regression discontinuity helpers and estimators."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import pandas.api.types as ptypes


@dataclass(frozen=True)
class RDDResult:
    """Local linear RDD estimate output."""

    estimate: float
    cutoff: float
    bandwidth: float
    standard_error: float


def _validate_inputs(
    data: pd.DataFrame,
    running_col: str,
    outcome_col: str,
    treatment_col: str,
    cutoff: float,
    bandwidth: float,
) -> None:
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("data must not be empty.")

    if not isinstance(running_col, str):
        raise ValueError("running_col must be a string.")
    if not isinstance(outcome_col, str):
        raise ValueError("outcome_col must be a string.")
    if not isinstance(treatment_col, str):
        raise ValueError("treatment_col must be a string.")

    required = [running_col, outcome_col, treatment_col]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if data[required].isna().any().any():
        raise ValueError("Required columns must not contain missing values.")
    if not ptypes.is_numeric_dtype(data[running_col]):
        raise ValueError("running_col must be numeric.")
    if not ptypes.is_numeric_dtype(data[outcome_col]):
        raise ValueError("outcome_col must be numeric.")
    if not ptypes.is_numeric_dtype(data[treatment_col]):
        raise ValueError("treatment_col must be numeric.")

    if ptypes.is_bool_dtype(data[treatment_col]):
        raise ValueError("treatment_col must not be boolean; use 0/1 values.")

    running = data[running_col].to_numpy(dtype=float)
    outcome = data[outcome_col].to_numpy(dtype=float)
    treatment = data[treatment_col].to_numpy(dtype=float)
    if not np.all(np.isfinite(running)) or not np.all(np.isfinite(outcome)):
        raise ValueError("running_col and outcome_col must contain finite values.")
    if not np.all(np.isfinite(treatment)):
        raise ValueError("treatment_col must contain finite values.")

    treatment_values = {value for value in np.unique(treatment)}
    if not treatment_values.issubset({0.0, 1.0}):
        raise ValueError("treatment_col must be binary (0/1).")

    if not isinstance(bandwidth, int | float) or isinstance(bandwidth, bool):
        raise ValueError("bandwidth must be positive.")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    if not isinstance(cutoff, (int, float)) or isinstance(cutoff, bool):
        raise ValueError("cutoff must be a finite real number.")
    if not np.isfinite(cutoff):
        raise ValueError("cutoff must be a finite real number.")


def local_linear_rdd(
    data: pd.DataFrame,
    running_col: str = "running",
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    cutoff: float = 0.0,
    bandwidth: float = 1.0,
) -> RDDResult:
    """Estimate RDD treatment effect by local linear regression around a cutoff."""

    _validate_inputs(data, running_col, outcome_col, treatment_col, cutoff, bandwidth)

    running = data[running_col].to_numpy(dtype=float)
    centered = running - cutoff
    mask = np.abs(centered) <= bandwidth
    if not mask.any():
        raise ValueError("No observations are inside the chosen bandwidth.")

    subset = data.loc[mask].copy()
    x = subset[running_col].to_numpy(dtype=float) - cutoff
    t = subset[treatment_col].to_numpy(dtype=float)
    y = subset[outcome_col].to_numpy(dtype=float)

    interaction = x * t
    design = pd.DataFrame({"intercept": np.ones_like(y), "running": x, "treatment": t, "interaction": interaction})
    model = sm.OLS(y, design).fit()
    estimate = float(model.params["treatment"])
    std_error = float(model.bse["treatment"])

    return RDDResult(
        estimate=estimate,
        cutoff=float(cutoff),
        bandwidth=float(bandwidth),
        standard_error=std_error,
    )


def rdd_bandwidth_sensitivity(
    data: pd.DataFrame,
    bandwidth_grid: Sequence[float],
    running_col: str = "running",
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    cutoff: float = 0.0,
) -> pd.DataFrame:
    """Evaluate local linear estimates across a bandwidth grid."""

    try:
        bandwidth_list = list(bandwidth_grid)
    except TypeError as exc:
        raise TypeError("bandwidth_grid must be a sequence of numeric bandwidths.") from exc

    if len(bandwidth_list) == 0:
        raise ValueError("bandwidth_grid must not be empty.")
    if any(isinstance(bandwidth, bool) or not isinstance(bandwidth, Real) for bandwidth in bandwidth_list):
        raise TypeError("bandwidth_grid must be a sequence of numeric bandwidths.")

    bandwidths = [float(bandwidth) for bandwidth in bandwidth_list]
    if not all(np.isfinite(bandwidths)):
        raise ValueError("all bandwidths must be finite positive values.")
    if any(bandwidth <= 0 for bandwidth in bandwidths):
        raise ValueError("all bandwidths must be positive.")

    rows = []
    for bandwidth in bandwidths:
        result = local_linear_rdd(
            data=data,
            running_col=running_col,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            cutoff=cutoff,
            bandwidth=float(bandwidth),
        )
        rows.append(
            {
                "bandwidth": result.bandwidth,
                "estimate": result.estimate,
                "standard_error": result.standard_error,
            }
        )
    return pd.DataFrame(rows)
