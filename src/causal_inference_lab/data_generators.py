"""Synthetic data generators with known causal ground truth.

Synthetic data is useful in a causal inference portfolio because the true causal
effect is known. This allows us to test whether an estimator is behaving as expected.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticDataset:
    """Container for a synthetic causal dataset.

    Attributes:
        data: The generated observations.
        true_ate: The population average treatment effect used by the generator.
        description: Short explanation of the data generating process.
    """

    data: pd.DataFrame
    true_ate: float
    description: str


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Return the logistic transform of a numeric array."""

    return 1.0 / (1.0 + np.exp(-x))


def _validate_positive_int(value: int, name: str) -> None:
    """Validate that an integer parameter is strictly positive."""

    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_seed(seed: int) -> None:
    """Validate RNG seed type."""

    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError("seed must be an integer.")


def _validate_treated_unit(treated_unit: int, n_units: int) -> None:
    """Validate treated unit id for synthetic control data."""

    if isinstance(treated_unit, bool) or not isinstance(treated_unit, numbers.Integral):
        raise TypeError("treated_unit must be an integer.")
    if treated_unit < 0:
        raise ValueError("treated_unit must be between 0 and n_units - 1.")
    if treated_unit >= n_units:
        raise ValueError("treated_unit must be between 0 and n_units - 1.")


def _validate_cutoff(cutoff: float) -> None:
    """Validate RDD cutoff scalar."""

    if isinstance(cutoff, bool) or not isinstance(cutoff, int | float):
        raise TypeError("cutoff must be a finite real number.")
    if not np.isfinite(cutoff):
        raise ValueError("cutoff must be finite.")


def make_confounded_binary_treatment(n: int = 5_000, seed: int = 42) -> SyntheticDataset:
    """Generate observational data with confounding and a binary treatment.

    The same covariates affect treatment assignment and the outcome. A naive comparison
    between treated and untreated units is therefore biased.

    Args:
        n: Number of observations.
        seed: Random seed.

    Returns:
        SyntheticDataset with treatment, outcome, covariates, propensity, and true ITE.
    """

    _validate_positive_int(n, "n")
    _validate_seed(seed)
    rng = np.random.default_rng(seed)

    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    x3 = rng.binomial(1, 0.45, n)

    propensity = _sigmoid(-0.2 + 0.9 * x1 - 0.7 * x2 + 0.5 * x3)
    treatment = rng.binomial(1, propensity)

    true_ite = 2.0 + 0.5 * x1
    baseline = 1.0 + 1.4 * x1 - 1.0 * x2 + 0.6 * x3
    noise = rng.normal(0.0, 1.0, n)
    outcome = baseline + true_ite * treatment + noise

    data = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "treatment": treatment,
            "outcome": outcome,
            "true_ite": true_ite,
            "true_propensity": propensity,
        }
    )

    return SyntheticDataset(
        data=data,
        true_ate=float(np.mean(true_ite)),
        description="Confounded binary treatment with heterogeneous treatment effects.",
    )


def make_heterogeneous_treatment_data(n: int = 5_000, seed: int = 123) -> SyntheticDataset:
    """Generate data where treatment effects vary strongly across units.

    Args:
        n: Number of observations.
        seed: Random seed.

    Returns:
        SyntheticDataset with known individual treatment effects.
    """

    _validate_positive_int(n, "n")
    _validate_seed(seed)
    rng = np.random.default_rng(seed)

    age = rng.normal(50.0, 12.0, n)
    risk_score = rng.normal(0.0, 1.0, n)
    prior_usage = rng.gamma(shape=2.0, scale=1.0, size=n)

    propensity = _sigmoid(-0.4 + 0.03 * (age - 50.0) + 0.8 * risk_score - 0.2 * prior_usage)
    treatment = rng.binomial(1, propensity)

    true_ite = 1.0 + 1.2 * (risk_score > 0.5).astype(float) + 0.02 * (age - 50.0)
    baseline = 5.0 + 0.04 * age + 1.5 * risk_score + 0.5 * prior_usage
    outcome = baseline + true_ite * treatment + rng.normal(0.0, 1.0, n)

    data = pd.DataFrame(
        {
            "age": age,
            "risk_score": risk_score,
            "prior_usage": prior_usage,
            "treatment": treatment,
            "outcome": outcome,
            "true_ite": true_ite,
            "true_propensity": propensity,
            "high_risk": (risk_score > 0.5).astype(int),
        }
    )

    return SyntheticDataset(
        data=data,
        true_ate=float(np.mean(true_ite)),
        description="Heterogeneous treatment effects by risk profile and age.",
    )


def make_did_panel(n_units: int = 600, n_periods: int = 8, seed: int = 7) -> SyntheticDataset:
    """Generate panel data for a difference-in-differences example.

    Treatment starts after the midpoint for the treated group. The data generating process
    satisfies parallel trends by construction.

    Args:
        n_units: Number of panel units.
        n_periods: Number of time periods.
        seed: Random seed.

    Returns:
        SyntheticDataset with unit, time, treated group, post indicator, and outcome.
    """

    _validate_positive_int(n_units, "n_units")
    _validate_positive_int(n_periods, "n_periods")
    _validate_seed(seed)
    if n_periods < 4:
        raise ValueError("n_periods must be at least 4 for a useful DiD example.")

    rng = np.random.default_rng(seed)
    effect: Final[float] = 3.0
    intervention_time = n_periods // 2

    treated_group = rng.binomial(1, 0.5, n_units)
    unit_effect = rng.normal(0.0, 1.0, n_units)

    rows: list[dict[str, float | int]] = []
    for unit in range(n_units):
        for time in range(n_periods):
            post = int(time >= intervention_time)
            treatment = int(treated_group[unit] == 1 and post == 1)
            time_effect = 0.5 * time
            outcome = (
                10.0
                + unit_effect[unit]
                + time_effect
                + 0.2 * treated_group[unit]
                + effect * treatment
                + rng.normal(0.0, 0.8)
            )
            rows.append(
                {
                    "unit": unit,
                    "time": time,
                    "treated_group": int(treated_group[unit]),
                    "post": post,
                    "treatment": treatment,
                    "outcome": outcome,
                }
            )

    return SyntheticDataset(
        data=pd.DataFrame(rows),
        true_ate=effect,
        description="Panel data satisfying parallel trends with a known treatment effect.",
    )


def make_iv_data(n: int = 5_000, seed: int = 99) -> SyntheticDataset:
    """Generate data for an instrumental variables example.

    The instrument affects treatment uptake but has no direct effect on the outcome.
    An unobserved confounder affects both treatment and outcome.

    Args:
        n: Number of observations.
        seed: Random seed.

    Returns:
        SyntheticDataset with instrument, treatment, outcome, and hidden confounder.
    """

    _validate_positive_int(n, "n")
    _validate_seed(seed)
    rng = np.random.default_rng(seed)

    u = rng.normal(0.0, 1.0, n)
    z = rng.binomial(1, 0.5, n)
    x = rng.normal(0.0, 1.0, n)

    treatment_prob = _sigmoid(-0.2 + 1.3 * z + 0.8 * u + 0.3 * x)
    treatment = rng.binomial(1, treatment_prob)

    effect: Final[float] = 2.5
    outcome = 1.0 + effect * treatment + 1.5 * u + 0.5 * x + rng.normal(0.0, 1.0, n)

    data = pd.DataFrame(
        {
            "instrument": z,
            "treatment": treatment,
            "outcome": outcome,
            "x": x,
            "hidden_confounder": u,
            "treatment_probability": treatment_prob,
        }
    )

    return SyntheticDataset(
        data=data,
        true_ate=effect,
        description="Instrumental variables data with unobserved confounding.",
    )


def make_sharp_rdd_data(n: int = 5_000, cutoff: float = 0.0, seed: int = 21) -> SyntheticDataset:
    """Generate a sharp regression discontinuity design dataset.

    The treatment is assigned mechanically as `running >= cutoff`.
    The true treatment effect is a known discontinuity at the cutoff.

    Args:
        n: Number of observations.
        cutoff: Threshold for treatment assignment.
        seed: Random seed.

    Returns:
        SyntheticDataset with running variable, treatment, outcome, and true ATE.
    """

    _validate_positive_int(n, "n")
    _validate_seed(seed)
    _validate_cutoff(cutoff)

    rng = np.random.default_rng(seed)
    running = rng.uniform(-3.0, 3.0, n)
    treatment = (running >= cutoff).astype(int)

    base = 5.0 + 0.8 * np.sin(running)
    effect = 2.5
    outcome = base + effect * treatment + 0.4 * running + rng.normal(0.0, 1.0, n)

    data = pd.DataFrame(
        {
            "running": running,
            "treatment": treatment,
            "outcome": outcome,
        }
    )
    return SyntheticDataset(
        data=data,
        true_ate=effect,
        description="Sharp RDD data with treatment assigned by running variable cutoff.",
    )


def make_synthetic_control_data(
    n_units: int = 40,
    n_periods: int = 16,
    pre_periods: int = 8,
    treated_unit: int = 0,
    effect: float = 2.0,
    seed: int = 101,
) -> SyntheticDataset:
    """Generate panel data for a one-treated-unit synthetic control example."""

    _validate_positive_int(n_units, "n_units")
    _validate_positive_int(n_periods, "n_periods")
    _validate_positive_int(pre_periods, "pre_periods")
    _validate_seed(seed)
    _validate_treated_unit(treated_unit, n_units=n_units)
    if pre_periods >= n_periods:
        raise ValueError("pre_periods must be less than n_periods.")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []

    unit_effects = rng.normal(0.0, 1.0, n_units)
    for unit in range(n_units):
        for time in range(n_periods):
            treatment = int(unit == treated_unit and time >= pre_periods)
            outcome = (
                3.0
                + unit_effects[unit]
                + 0.25 * time
                + effect * treatment
                + rng.normal(0.0, 0.8)
            )
            rows.append(
                {
                    "unit": unit,
                    "time": time,
                    "treated_unit": int(unit == treated_unit),
                    "treatment": treatment,
                    "outcome": outcome,
                }
            )

    return SyntheticDataset(
        data=pd.DataFrame(rows),
        true_ate=effect,
        description="Panel data with one treated unit and many donor units for synthetic control.",
    )
