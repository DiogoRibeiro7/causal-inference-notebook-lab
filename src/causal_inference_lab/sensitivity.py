"""Simple sensitivity checks for causal analyses."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Real
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from causal_inference_lab.estimators import EffectEstimate


@dataclass(frozen=True)
class SensitivityResult:
    """Result from a sensitivity check."""

    name: str
    estimate: float
    reference_estimate: float
    interpretation: str


def _validate_data_frame(data: Any) -> None:
    """Validate `data` is a non-empty pandas DataFrame."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("data must not be empty.")


def _validate_covariates(data: pd.DataFrame, covariates: Sequence[str]) -> list[str]:
    """Validate a covariate column sequence and return it as a list."""

    if isinstance(covariates, str):
        raise TypeError("covariates must be a sequence of column names, not a string.")
    try:
        covariates_list = list(covariates)
    except TypeError as exc:
        raise TypeError("covariates must be a sequence of column names.") from exc

    if not all(isinstance(covariate, str) for covariate in covariates_list):
        raise TypeError("covariates must be a sequence of column names.")

    missing = [covariate for covariate in covariates_list if covariate not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return covariates_list


def _validate_treatment_col(data: pd.DataFrame, treatment_col: str) -> np.ndarray:
    """Validate treatment column and return numeric binary treatment values."""

    if not isinstance(treatment_col, str):
        raise TypeError("treatment_col must be a string.")
    if treatment_col not in data.columns:
        raise ValueError(f"Unknown treatment column: {treatment_col}")

    try:
        treatment = data[treatment_col].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("treatment must be numeric and finite.") from exc

    if not np.all(np.isfinite(treatment)):
        raise ValueError("Treatment values must be finite.")
    if not np.isin(np.unique(treatment), [0, 1]).all():
        raise ValueError("treatment must be binary and encoded as 0/1.")

    return treatment


def _validate_seed(seed: int) -> None:
    """Validate RNG seed value."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")


def _validate_numeric_value(value: Any, name: str) -> float:
    """Validate a scalar numeric input and return it as float."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a numeric value.")
    value_float = float(value)
    if not np.isfinite(value_float):
        raise ValueError(f"{name} must be finite.")
    return value_float


def _validate_confounder_strength_grid(strength_grid: Sequence[float]) -> list[float]:
    """Validate confounder-strength grid input."""

    if isinstance(strength_grid, str):
        raise TypeError("confounder_strength_grid must be a sequence of numeric values.")
    try:
        strengths = list(strength_grid)
    except TypeError as exc:
        raise TypeError("confounder_strength_grid must be a sequence of numeric values.") from exc
    if not strengths:
        raise ValueError("confounder_strength_grid must not be empty.")

    numeric_strengths = []
    for strength in strengths:
        if isinstance(strength, bool) or not isinstance(strength, Real):
            raise ValueError("confounder_strength_grid must contain numeric values.")
        strength_float = float(strength)
        if not np.isfinite(strength_float):
            raise ValueError("confounder_strength_grid values must be finite.")
        if strength_float < 0:
            raise ValueError("confounder_strength_grid values must be non-negative.")
        numeric_strengths.append(strength_float)

    return numeric_strengths


def _estimate_treatment_effect(
    data: pd.DataFrame,
    estimator: Callable[[pd.DataFrame, Sequence[str]], EffectEstimate],
    covariates: Sequence[str],
) -> EffectEstimate:
    """Run an estimator and enforce its output type."""

    estimate = estimator(data, covariates)
    if not isinstance(estimate, EffectEstimate):
        raise TypeError("estimator must return an EffectEstimate.")
    if not np.isfinite(estimate.estimate):
        raise ValueError("estimator produced a non-finite estimate.")
    return estimate


def placebo_treatment_test(
    data: pd.DataFrame,
    estimator: Callable[[pd.DataFrame, Sequence[str]], EffectEstimate],
    covariates: Sequence[str],
    treatment_col: str = "treatment",
    seed: int = 123,
) -> SensitivityResult:
    """Shuffle treatment assignment and re-estimate the effect.

    A large placebo effect can indicate estimator instability, poor overlap, or a design
    that is too sensitive to the treatment assignment mechanism.
    """
    if not callable(estimator):
        raise TypeError("estimator must be callable.")

    _validate_data_frame(data)
    covariates_list = _validate_covariates(data, covariates)
    _validate_treatment_col(data, treatment_col)
    _validate_seed(seed)

    rng = np.random.default_rng(seed)
    placebo_data = data.copy()
    treatment = _validate_treatment_col(data, treatment_col)
    placebo_data[treatment_col] = rng.permutation(treatment)

    reference = _estimate_treatment_effect(data, estimator, covariates_list)
    placebo = _estimate_treatment_effect(placebo_data, estimator, covariates_list)

    return SensitivityResult(
        name="placebo_treatment_shuffle",
        estimate=placebo.estimate,
        reference_estimate=reference.estimate,
        interpretation=(
            "The placebo estimate should be close to zero. If it is not, inspect overlap, "
            "functional form, and data leakage."
        ),
    )


def omitted_confounder_simulation(
    data: pd.DataFrame,
    base_effect: float,
    confounder_strength_grid: Sequence[float],
    treatment_col: str = "treatment",
    seed: int = 123,
) -> pd.DataFrame:
    """Simulate how an omitted confounder could shift an effect estimate.

    This is a simple educational sensitivity analysis, not a replacement for a full
    domain-specific sensitivity model.
    """
    _validate_data_frame(data)
    treatment = _validate_treatment_col(data, treatment_col)
    base_effect = _validate_numeric_value(base_effect, "base_effect")
    strengths = _validate_confounder_strength_grid(confounder_strength_grid)
    _validate_seed(seed)
    if np.var(treatment) == 0:
        raise ValueError("Treatment must have variation for sensitivity analysis.")

    treatment_centered = treatment - np.mean(treatment)

    rows = []
    for strength in strengths:
        hidden = strength * treatment_centered
        confounder_effect = strength

        treatment_covariance = float(np.cov(treatment, hidden, ddof=0)[0, 1])
        variance_treatment = float(np.var(treatment, ddof=0))
        if variance_treatment == 0.0:
            raise ValueError("Treatment variance is zero; confounding sensitivity is undefined.")

        bias = float(confounder_effect * treatment_covariance / variance_treatment)
        rows.append(
            {
                "confounder_strength": float(strength),
                "simulated_bias": bias,
                "adjusted_effect": float(base_effect - bias),
            }
        )

    return pd.DataFrame(rows)
