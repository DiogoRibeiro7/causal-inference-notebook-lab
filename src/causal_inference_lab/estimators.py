"""Core causal effect estimators used by the notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import RegressorMixin
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.utils.validation import check_is_fitted


@dataclass(frozen=True)
class EffectEstimate:
    """A causal effect estimate with minimal metadata."""

    estimate: float
    estimator: str
    estimand: str
    n_observations: int


def _validate_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    """Validate that all expected columns exist in a DataFrame."""

    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _as_numpy_frame(data: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Return selected columns as a numeric NumPy array."""

    _validate_columns(data, columns)
    return data.loc[:, list(columns)].to_numpy(dtype=float)


def _validate_binary_treatment(treatment: pd.Series) -> None:
    """Validate that treatment contains only 0 and 1."""

    values = set(treatment.dropna().unique().tolist())
    if not values.issubset({0, 1}):
        raise ValueError("Treatment must be binary and encoded as 0/1.")


def difference_in_means(
    data: pd.DataFrame,
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
) -> EffectEstimate:
    """Estimate a naive difference in mean outcomes.

    This is not a causal estimator in observational data unless treatment is as-if random.

    Args:
        data: Input data.
        treatment_col: Binary treatment column.
        outcome_col: Outcome column.

    Returns:
        EffectEstimate for treated mean minus untreated mean.
    """

    _validate_columns(data, [treatment_col, outcome_col])
    _validate_binary_treatment(data[treatment_col])

    treated = data.loc[data[treatment_col] == 1, outcome_col]
    control = data.loc[data[treatment_col] == 0, outcome_col]

    if treated.empty or control.empty:
        raise ValueError("Both treated and control groups must contain observations.")

    return EffectEstimate(
        estimate=float(treated.mean() - control.mean()),
        estimator="difference_in_means",
        estimand="ATE",
        n_observations=int(len(data)),
    )


def fit_propensity_model(
    data: pd.DataFrame,
    covariates: Sequence[str],
    treatment_col: str = "treatment",
) -> LogisticRegression:
    """Fit a logistic regression propensity score model."""

    _validate_columns(data, [treatment_col, *covariates])
    _validate_binary_treatment(data[treatment_col])

    model = LogisticRegression(max_iter=1_000)
    model.fit(_as_numpy_frame(data, covariates), data[treatment_col].to_numpy(dtype=int))
    return model


def estimate_propensity_scores(
    data: pd.DataFrame,
    covariates: Sequence[str],
    treatment_col: str = "treatment",
    clip: float = 0.01,
) -> np.ndarray:
    """Estimate propensity scores with logistic regression.

    Args:
        data: Input data.
        covariates: Covariate names used in the treatment model.
        treatment_col: Binary treatment column.
        clip: Lower and upper clipping value to avoid extreme weights.

    Returns:
        Array of clipped propensity scores.
    """

    if not 0.0 < clip < 0.5:
        raise ValueError("clip must be between 0 and 0.5.")

    model = fit_propensity_model(data, covariates, treatment_col)
    scores = model.predict_proba(_as_numpy_frame(data, covariates))[:, 1]
    return np.clip(scores, clip, 1.0 - clip)


def ipw_ate(
    data: pd.DataFrame,
    covariates: Sequence[str],
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    clip: float = 0.01,
) -> EffectEstimate:
    """Estimate the ATE with inverse probability weighting."""

    _validate_columns(data, [treatment_col, outcome_col, *covariates])
    _validate_binary_treatment(data[treatment_col])

    treatment = data[treatment_col].to_numpy(dtype=float)
    outcome = data[outcome_col].to_numpy(dtype=float)
    propensity = estimate_propensity_scores(data, covariates, treatment_col, clip=clip)

    treated_component = treatment * outcome / propensity
    control_component = (1.0 - treatment) * outcome / (1.0 - propensity)
    estimate = float(np.mean(treated_component - control_component))

    return EffectEstimate(
        estimate=estimate,
        estimator="inverse_probability_weighting",
        estimand="ATE",
        n_observations=int(len(data)),
    )


def g_computation_ate(
    data: pd.DataFrame,
    covariates: Sequence[str],
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    outcome_model: RegressorMixin | None = None,
) -> EffectEstimate:
    """Estimate the ATE using outcome regression / g-computation."""

    _validate_columns(data, [treatment_col, outcome_col, *covariates])
    _validate_binary_treatment(data[treatment_col])

    model = outcome_model or LinearRegression()
    features = list(covariates) + [treatment_col]
    model.fit(_as_numpy_frame(data, features), data[outcome_col].to_numpy(dtype=float))

    treated_frame = data.copy()
    control_frame = data.copy()
    treated_frame[treatment_col] = 1
    control_frame[treatment_col] = 0

    mu1 = model.predict(_as_numpy_frame(treated_frame, features))
    mu0 = model.predict(_as_numpy_frame(control_frame, features))

    return EffectEstimate(
        estimate=float(np.mean(mu1 - mu0)),
        estimator="g_computation",
        estimand="ATE",
        n_observations=int(len(data)),
    )


def aipw_ate(
    data: pd.DataFrame,
    covariates: Sequence[str],
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    clip: float = 0.01,
) -> EffectEstimate:
    """Estimate the ATE using the augmented inverse probability weighting estimator.

    AIPW combines an outcome model and a treatment model. Under regularity conditions,
    it is consistent if either the outcome model or the propensity model is correctly
    specified.
    """

    _validate_columns(data, [treatment_col, outcome_col, *covariates])
    _validate_binary_treatment(data[treatment_col])

    treatment = data[treatment_col].to_numpy(dtype=float)
    outcome = data[outcome_col].to_numpy(dtype=float)

    propensity = estimate_propensity_scores(data, covariates, treatment_col, clip=clip)

    features = list(covariates) + [treatment_col]
    outcome_model = LinearRegression()
    outcome_model.fit(_as_numpy_frame(data, features), outcome)

    treated_frame = data.copy()
    control_frame = data.copy()
    treated_frame[treatment_col] = 1
    control_frame[treatment_col] = 0

    mu1 = outcome_model.predict(_as_numpy_frame(treated_frame, features))
    mu0 = outcome_model.predict(_as_numpy_frame(control_frame, features))

    correction_treated = treatment * (outcome - mu1) / propensity
    correction_control = (1.0 - treatment) * (outcome - mu0) / (1.0 - propensity)
    pseudo_outcome = mu1 - mu0 + correction_treated - correction_control

    return EffectEstimate(
        estimate=float(np.mean(pseudo_outcome)),
        estimator="augmented_inverse_probability_weighting",
        estimand="ATE",
        n_observations=int(len(data)),
    )


def ols_treatment_effect(
    data: pd.DataFrame,
    covariates: Sequence[str],
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
) -> EffectEstimate:
    """Estimate a treatment coefficient with an OLS adjustment model."""

    _validate_columns(data, [treatment_col, outcome_col, *covariates])
    features = sm.add_constant(data.loc[:, [treatment_col, *covariates]], has_constant="add")
    model = sm.OLS(data[outcome_col], features).fit()

    return EffectEstimate(
        estimate=float(model.params[treatment_col]),
        estimator="ols_adjusted",
        estimand="ATE under linear-model assumptions",
        n_observations=int(len(data)),
    )


def predict_cate_t_learner(
    data: pd.DataFrame,
    covariates: Sequence[str],
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    model_treated: RegressorMixin | None = None,
    model_control: RegressorMixin | None = None,
) -> np.ndarray:
    """Estimate CATE with a simple T-learner.

    Separate outcome models are fitted for treated and control units. The predicted CATE
    is the difference between the two predicted potential outcomes.
    """

    _validate_columns(data, [treatment_col, outcome_col, *covariates])
    _validate_binary_treatment(data[treatment_col])

    treated_data = data[data[treatment_col] == 1]
    control_data = data[data[treatment_col] == 0]
    if treated_data.empty or control_data.empty:
        raise ValueError("Both treated and control groups are required.")

    treated_model = model_treated or LinearRegression()
    control_model = model_control or LinearRegression()

    treated_model.fit(_as_numpy_frame(treated_data, covariates), treated_data[outcome_col])
    control_model.fit(_as_numpy_frame(control_data, covariates), control_data[outcome_col])

    check_is_fitted(treated_model)
    check_is_fitted(control_model)

    x = _as_numpy_frame(data, covariates)
    return treated_model.predict(x) - control_model.predict(x)
