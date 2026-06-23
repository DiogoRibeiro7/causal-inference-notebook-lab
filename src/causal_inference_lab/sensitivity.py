"""Simple sensitivity checks for causal analyses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Sequence

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

    if treatment_col not in data.columns:
        raise ValueError(f"Unknown treatment column: {treatment_col}")

    rng = np.random.default_rng(seed)
    placebo_data = data.copy()
    placebo_data[treatment_col] = rng.permutation(placebo_data[treatment_col].to_numpy())

    reference = estimator(data, covariates)
    placebo = estimator(placebo_data, covariates)

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

    if treatment_col not in data.columns:
        raise ValueError(f"Unknown treatment column: {treatment_col}")

    if not confounder_strength_grid:
        raise ValueError("confounder_strength_grid must not be empty.")

    treatment = data[treatment_col].to_numpy(dtype=float)
    if np.var(treatment) == 0:
        raise ValueError("Treatment must have variation for sensitivity analysis.")
    if np.any(~np.isfinite(treatment)):
        raise ValueError("Treatment values must be finite.")

    treatment_centered = treatment - np.mean(treatment)

    rows = []
    for strength in confounder_strength_grid:
        if not isinstance(strength, (int, float)):
            raise ValueError("confounder_strength_grid must contain numeric values.")
        strength = float(strength)
        if strength < 0:
            raise ValueError("confounder_strength_grid values must be non-negative.")

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
