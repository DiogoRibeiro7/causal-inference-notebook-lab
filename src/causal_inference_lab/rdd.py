"""Regression discontinuity helpers and estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


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
    required = [running_col, outcome_col, treatment_col]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not isinstance(bandwidth, int | float) or bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    if not np.isfinite(cutoff):
        raise ValueError("cutoff must be finite.")


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

    if not bandwidth_grid:
        raise ValueError("bandwidth_grid must not be empty.")
    if any(float(bw) <= 0 for bw in bandwidth_grid):
        raise ValueError("all bandwidths must be positive.")

    rows = []
    for bandwidth in bandwidth_grid:
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
