"""Double machine learning estimators for causal effects."""

from __future__ import annotations

from typing import Sequence

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
) -> None:
    missing = [column for column in [treatment_col, outcome_col, *covariates] if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    treatment_values = set(data[treatment_col].dropna().unique().tolist())
    if not treatment_values.issubset({0, 1}):
        raise ValueError("treatment must be binary and encoded as 0/1.")


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

    _validate_data_inputs(data, covariates, treatment_col, outcome_col)
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2.")

    x = data.loc[:, list(covariates)].to_numpy(dtype=float)
    d = data[treatment_col].to_numpy(dtype=float)
    y = data[outcome_col].to_numpy(dtype=float)

    outcome_base = outcome_model or LinearRegression()
    treatment_base = treatment_model or LogisticRegression(max_iter=1_000)

    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    residual_t: list[float] = []
    residual_y: list[float] = []

    for train_idx, test_idx in folds.split(x):
        x_train = x[train_idx]
        y_train = y[train_idx]
        d_train = d[train_idx]
        x_test = x[test_idx]
        d_test = d[test_idx]
        y_test = y[test_idx]

        outcome_fold = clone(outcome_base)
        treatment_fold = clone(treatment_base)

        outcome_features_train = np.column_stack((x_train, d_train))
        outcome_features_test = np.column_stack((x_test, d_test))
        outcome_fold.fit(outcome_features_train, y_train)
        y_hat = outcome_fold.predict(outcome_features_test)

        treatment_fold.fit(x_train, d_train)
        e_hat = treatment_fold.predict_proba(x_test)[:, 1]

        e_hat = np.clip(e_hat, 1e-3, 1 - 1e-3)
        residual_t.extend((d_test - e_hat).tolist())
        residual_y.extend((y_test - y_hat).tolist())

    residual_t_arr = np.asarray(residual_t, dtype=float)
    residual_y_arr = np.asarray(residual_y, dtype=float)
    denominator = float(np.dot(residual_t_arr, residual_t_arr))
    if denominator == 0.0:
        raise ValueError("Residualized treatment variance is zero; check overlap and model specification.")

    ate = float(np.dot(residual_t_arr, residual_y_arr) / denominator)

    return EffectEstimate(
        estimate=ate,
        estimator="double_machine_learning",
        estimand="ATE",
        n_observations=int(len(data)),
    )
