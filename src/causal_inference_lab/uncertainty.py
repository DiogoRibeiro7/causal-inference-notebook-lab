"""Reusable uncertainty utilities for estimator output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from causal_inference_lab.estimators import EffectEstimate


@dataclass(frozen=True)
class BootstrapResult:
    """Result from a bootstrap uncertainty calculation."""

    estimate: float
    lower: float
    upper: float
    std_error: float
    n_bootstrap_samples: int
    confidence_level: float
    n_observations: int


def _validate_common_inputs(
    data: pd.DataFrame,
    n_bootstrap_samples: int,
    confidence_level: float,
) -> None:
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("data must not be empty.")
    if not isinstance(n_bootstrap_samples, int) or n_bootstrap_samples <= 0:
        raise ValueError("n_bootstrap_samples must be a positive integer.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be strictly between 0 and 1.")


def bootstrap_ate(
    data: pd.DataFrame,
    estimator: Callable[[pd.DataFrame], EffectEstimate] | Callable[[pd.DataFrame, Sequence[str]], EffectEstimate],
    n_bootstrap_samples: int = 1_000,
    seed: int = 123,
    confidence_level: float = 0.95,
    covariates: Sequence[str] | None = None,
) -> BootstrapResult:
    """Compute a bootstrap interval and standard error for a causal estimate.

    The estimator must return an :class:`EffectEstimate`.

    Args:
        data: Input dataset for resampling.
        estimator: Estimator callable that accepts a DataFrame, and optionally covariates.
        n_bootstrap_samples: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.
        confidence_level: Confidence level for interval bounds.
        covariates: Optional covariates passed to estimators that require them.

    Returns:
        BootstrapResult with point estimate, interval bounds, standard error, and sample count.
    """

    _validate_common_inputs(data, n_bootstrap_samples, confidence_level)
    if n_bootstrap_samples < 2:
        raise ValueError("n_bootstrap_samples must be at least 2 to estimate a standard error.")
    if covariates is not None:
        if not isinstance(covariates, Sequence):
            raise ValueError("covariates must be a sequence of column names.")
        if not covariates:
            raise ValueError("covariates must not be empty when provided.")

    rng = np.random.default_rng(seed)
    bootstrap_estimates = np.empty(n_bootstrap_samples, dtype=float)

    if covariates is None:
        base_estimate = float(estimator(data).estimate)
        for i in range(n_bootstrap_samples):
            sample_index = rng.choice(data.index, size=len(data), replace=True)
            sample = data.loc[sample_index].reset_index(drop=True)
            bootstrap_estimates[i] = float(estimator(sample).estimate)
    else:
        base_estimate = float(estimator(data, covariates).estimate)
        for i in range(n_bootstrap_samples):
            sample_index = rng.choice(data.index, size=len(data), replace=True)
            sample = data.loc[sample_index].reset_index(drop=True)
            bootstrap_estimates[i] = float(estimator(sample, covariates).estimate)

    alpha = 1.0 - confidence_level
    lower_tail = alpha / 2.0
    upper_tail = 1.0 - lower_tail

    return BootstrapResult(
        estimate=base_estimate,
        lower=float(np.quantile(bootstrap_estimates, lower_tail)),
        upper=float(np.quantile(bootstrap_estimates, upper_tail)),
        std_error=float(np.std(bootstrap_estimates, ddof=1)),
        n_bootstrap_samples=n_bootstrap_samples,
        confidence_level=confidence_level,
        n_observations=int(len(data)),
    )
