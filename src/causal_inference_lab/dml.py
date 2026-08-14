"""Double machine learning estimators for causal effects."""

from __future__ import annotations

import numbers
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

from causal_inference_lab.estimators import EffectEstimate


def _validate_data_inputs(
    data: pd.DataFrame,
    covariates: Sequence[str],
    treatment_col: str,
    outcome_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("data must not be empty.")

    if not isinstance(treatment_col, str):
        raise TypeError("treatment_col must be a string.")
    if not isinstance(outcome_col, str):
        raise TypeError("outcome_col must be a string.")
    if treatment_col == outcome_col:
        raise ValueError("treatment_col and outcome_col must be different.")

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

    missing = [
        column
        for column in [treatment_col, outcome_col, *covariate_list]
        if column not in data.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    treatment_values = set(data[treatment_col].dropna().unique().tolist())
    if not treatment_values.issubset({0, 1}):
        raise ValueError("treatment must be binary and encoded as 0/1.")
    if not np.all(np.isin(list(treatment_values), [0, 1])):
        raise ValueError("treatment must be binary and encoded as 0/1.")

    try:
        x = data.loc[:, covariate_list].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("covariates must be numeric and finite.") from exc
    if not np.all(np.isfinite(x)):
        raise ValueError("covariates must not contain NaN or infinite values.")

    try:
        y = data[outcome_col].astype(float).to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("outcome must be numeric and finite.") from exc
    if not np.isfinite(y).all():
        raise ValueError("outcome must not contain NaN or infinite values.")

    t = data[treatment_col].astype(float).to_numpy(dtype=float)
    if not np.all(np.isfinite(t)):
        raise ValueError("treatment must not contain NaN or infinite values.")
    if np.any(np.isin(t, [0, 1], invert=True)):
        raise ValueError("treatment must be binary and encoded as 0/1.")

    if not np.any(t == 1) or not np.any(t == 0):
        raise ValueError("Both treated and control groups are required.")

    return x, t, y


def double_machine_learning_ate(
    data: pd.DataFrame,
    covariates: Sequence[str],
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    n_splits: int = 5,
    seed: int = 123,
    outcome_model: RegressorMixin | None = None,
    treatment_model: LogisticRegression | None = None,
) -> EffectEstimate:
    """Estimate a constant treatment effect with cross-fitted nuisance models.

    This implementation uses cross-fitting to avoid bias from overfitting the nuisance models
    when estimating the final score function.

    Args:
        data: Input data.
        covariates: Covariate names for nuisance models.
        treatment_col: Binary treatment column.
        outcome_col: Outcome column.
        n_splits: Number of cross-fitting folds.
        seed: Random seed for fold assignment.
        outcome_model: Optional outcome nuisance model.
        treatment_model: Optional treatment propensity model.

    Returns:
        EffectEstimate with the DML ATE estimate.
    """

    x, treatment, outcome = _validate_data_inputs(
        data,
        covariates,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
    )
    if isinstance(n_splits, bool) or not isinstance(n_splits, numbers.Integral) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2.")
    n_splits = int(n_splits)
    if n_splits > len(data):
        raise ValueError("n_splits cannot exceed the number of observations.")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError("seed must be an integer.")
    seed = int(seed)

    outcome_base = outcome_model if outcome_model is not None else LinearRegression()
    treatment_base = (
        treatment_model if treatment_model is not None else LogisticRegression(max_iter=1_000)
    )

    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    residual_t: list[float] = []
    residual_y: list[float] = []

    for train_idx, test_idx in folds.split(x):
        x_train = x[train_idx]
        y_train = outcome[train_idx]
        d_train = treatment[train_idx]
        x_test = x[test_idx]
        d_test = treatment[test_idx]
        y_test = outcome[test_idx]

        outcome_fold = clone(outcome_base)
        treatment_fold = clone(treatment_base)

        # Robinson partialling-out: the outcome nuisance estimates E[Y | X] only.
        # Including treatment here would absorb the effect we are trying to estimate.
        outcome_fold.fit(x_train, y_train)
        y_hat = outcome_fold.predict(x_test)

        treatment_fold.fit(x_train, d_train)
        e_hat = treatment_fold.predict_proba(x_test)[:, 1]

        e_hat = np.clip(e_hat, 1e-3, 1 - 1e-3)
        residual_t.extend((d_test - e_hat).tolist())
        residual_y.extend((y_test - y_hat).tolist())

    residual_t_arr = np.asarray(residual_t, dtype=float)
    residual_y_arr = np.asarray(residual_y, dtype=float)
    denominator = float(np.dot(residual_t_arr, residual_t_arr))
    if denominator == 0.0:
        raise ValueError(
            "Residualized treatment variance is zero; check overlap and model specification."
        )

    ate = float(np.dot(residual_t_arr, residual_y_arr) / denominator)

    return EffectEstimate(
        estimate=ate,
        estimator="double_machine_learning",
        estimand="ATE",
        n_observations=int(len(data)),
    )
