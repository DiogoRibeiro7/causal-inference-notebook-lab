"""Heterogeneous treatment effect meta-learners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression


def _ensure_dataframe(data: pd.DataFrame, *, name: str) -> None:
    """Validate that ``data`` is a non-empty DataFrame."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if data.empty:
        raise ValueError(f"{name} must contain at least one row.")


def _ensure_covariates(data: pd.DataFrame, covariates: Sequence[str], *, name: str = "data") -> list[str]:
    """Validate covariate column names and return them as a list."""

    if isinstance(covariates, str):
        raise TypeError(f"{name} covariates must be a sequence of column names, not a string.")

    try:
        covariate_list = list(covariates)
    except TypeError as exc:
        raise TypeError(f"{name} covariates must be a non-empty sequence of column names.") from exc

    if not covariate_list:
        raise ValueError(f"{name} covariates must not be empty.")
    if len(set(covariate_list)) != len(covariate_list):
        raise ValueError(f"{name} covariates must be unique.")

    missing = [column for column in covariate_list if column not in data.columns]
    if missing:
        raise ValueError(f"{name} is missing required covariate columns: {missing}.")

    return covariate_list


def _coerce_numeric_matrix(data: pd.DataFrame, *, name: str) -> np.ndarray:
    """Convert a DataFrame to a dense float matrix and validate it."""

    if data.isnull().any().any():
        raise ValueError(f"{name} contains missing values; remove or impute before fitting.")
    try:
        matrix = data.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc

    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    return matrix


def _coerce_binary_treatment(treatment: pd.Series, *, name: str) -> np.ndarray:
    """Validate treatment as binary 0/1 and convert to int array."""

    values = treatment.astype(float)
    if values.isnull().any():
        raise ValueError(f"{name} contains missing values.")

    numeric = values.to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} contains non-finite values.")
    if not np.array_equal(numeric, numeric.astype(int)):
        raise ValueError(f"{name} must be binary and encoded as 0/1.")
    if not np.isin(np.unique(numeric), [0.0, 1.0]).all():
        raise ValueError(f"{name} must be binary and encoded as 0/1.")

    return numeric.astype(int)


def _coerce_numeric_series(values: pd.Series, *, name: str) -> np.ndarray:
    """Convert a Series to float and validate it contains finite values."""

    numeric = values.astype(float)
    if numeric.isnull().any():
        raise ValueError(f"{name} contains missing values.")

    array = numeric.to_numpy(dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    return array


def _prepare_meta_learning_data(
    data: pd.DataFrame,
    covariates: Sequence[str],
    treatment_col: str,
    outcome_col: str,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Validate inputs shared by all meta-learners and return cleaned arrays."""

    _ensure_dataframe(data, name="data")
    if not isinstance(treatment_col, str):
        raise TypeError("treatment_col must be a string.")
    if not isinstance(outcome_col, str):
        raise TypeError("outcome_col must be a string.")
    if treatment_col == outcome_col:
        raise ValueError("treatment_col and outcome_col must be different.")

    covariate_list = _ensure_covariates(data, covariates, name="data")
    treatment = _coerce_binary_treatment(data[treatment_col], name="treatment")
    outcome = _coerce_numeric_series(data[outcome_col], name="outcome")
    x_matrix = _coerce_numeric_matrix(data.loc[:, covariate_list], name="covariates")

    return covariate_list, x_matrix, treatment, outcome


def _validate_prediction_frame(
    x: pd.DataFrame,
    covariates: Sequence[str],
    *,
    name: str,
) -> np.ndarray:
    """Validate prediction input and return a float matrix."""

    _ensure_dataframe(x, name=name)
    if not covariates:
        raise ValueError("Model is missing fitted covariates. Call fit(...) first.")

    missing = [column for column in covariates if column not in x.columns]
    if missing:
        raise ValueError(f"{name} is missing required covariate columns: {missing}.")

    return _coerce_numeric_matrix(x.loc[:, list(covariates)], name="x")


def _validate_group_overlap(treatment: np.ndarray, *, method: str) -> None:
    """Validate that both treatment groups are present."""

    if not np.any(treatment == 1) or not np.any(treatment == 0):
        raise ValueError(f"{method} requires both treated and control groups.")


class CATEModel(ABC):
    """Base class for CATE estimators."""

    def __init__(self) -> None:
        self._is_fitted = False
        self._covariates: list[str] = []
        self.estimator: RegressorMixin | None = None

    def _require_fitted(self) -> None:
        """Raise when calling prediction before fitting."""

        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit(...) first.")

    @abstractmethod
    def predict_cate(self, x: pd.DataFrame) -> np.ndarray:
        """Predict CATE for each row in ``x``."""

    def estimate_ate(self, x: pd.DataFrame) -> float:
        """Return the average predicted CATE."""
        self._require_fitted()
        return float(np.mean(self.predict_cate(x)))


class SMetaLearner(CATEModel):
    """S-learner: outcome model includes treatment as a feature."""

    def __init__(self, model: RegressorMixin | None = None) -> None:
        super().__init__()
        self.model = model or LinearRegression()

    def _features(self, x: np.ndarray, treatment: np.ndarray | None = None) -> np.ndarray:
        """Build S-learner features by appending treatment to each row."""

        if treatment is None:
            treatment = np.zeros(len(x), dtype=float)
        if len(x) != len(treatment):
            raise ValueError("x and treatment must contain the same number of rows.")

        return np.column_stack([x, treatment.astype(float)])

    def fit(self, data: pd.DataFrame, covariates: Sequence[str], treatment_col: str, outcome_col: str) -> "SMetaLearner":
        """Fit the S-learner outcome model.\n\n        Args:\n+            data: Input DataFrame.\n+            covariates: Covariate column names.\n+            treatment_col: Binary treatment column.\n+            outcome_col: Outcome column.\n\n        Returns:\n+            Self.\n        """
        covariate_list, x_matrix, treatment, outcome = _prepare_meta_learning_data(
            data,
            covariates,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
        )
        self._covariates = covariate_list
        self.model.fit(self._features(x_matrix, treatment), outcome)
        self._is_fitted = True
        return self

    def predict_cate(self, x: pd.DataFrame) -> np.ndarray:
        """Predict CATE via counterfactual predictions with/without treatment."""

        self._require_fitted()
        x_matrix = _validate_prediction_frame(x, self._covariates, name="x")
        treated = self._features(x_matrix, np.ones(len(x_matrix), dtype=float))
        untreated = self._features(x_matrix, np.zeros(len(x_matrix), dtype=float))
        return self.model.predict(treated) - self.model.predict(untreated)


class TMetaLearner(CATEModel):
    """T-learner: separate outcome models in treated and control groups."""

    def __init__(
        self,
        treated_model: RegressorMixin | None = None,
        control_model: RegressorMixin | None = None,
    ) -> None:
        super().__init__()
        self.treated_model = treated_model or LinearRegression()
        self.control_model = control_model or LinearRegression()

    def fit(self, data: pd.DataFrame, covariates: Sequence[str], treatment_col: str, outcome_col: str) -> "TMetaLearner":
        """Fit separate outcome models for treated and control units.\n\n        Args:\n            data: Input DataFrame.\n            covariates: Covariate column names.\n+            treatment_col: Binary treatment column.\n+            outcome_col: Outcome column.\n\n        Returns:\n            Self.\n        """
        covariate_list, x_matrix, treatment, outcome = _prepare_meta_learning_data(
            data,
            covariates,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
        )
        _validate_group_overlap(treatment, method="TMetaLearner")
        self._covariates = covariate_list
        self.treated_model.fit(x_matrix[treatment == 1], outcome[treatment == 1])
        self.control_model.fit(x_matrix[treatment == 0], outcome[treatment == 0])
        self._is_fitted = True
        return self

    def predict_cate(self, x: pd.DataFrame) -> np.ndarray:
        """Predict CATE from predicted treated minus predicted control outcomes."""

        self._require_fitted()
        x_matrix = _validate_prediction_frame(x, self._covariates, name="x")
        return self.treated_model.predict(x_matrix) - self.control_model.predict(x_matrix)


class XMetaLearner(CATEModel):
    """X-learner with simple stage-two effect models."""

    def __init__(
        self,
        outcome_model_t: RegressorMixin | None = None,
        outcome_model_c: RegressorMixin | None = None,
        effect_model_t: RegressorMixin | None = None,
        effect_model_c: RegressorMixin | None = None,
    ) -> None:
        super().__init__()
        self.outcome_model_t = outcome_model_t or LinearRegression()
        self.outcome_model_c = outcome_model_c or LinearRegression()
        self.effect_model_t = effect_model_t or LinearRegression()
        self.effect_model_c = effect_model_c or LinearRegression()
        self.ate_: float | None = None

    def fit(self, data: pd.DataFrame, covariates: Sequence[str], treatment_col: str, outcome_col: str) -> "XMetaLearner":
        """Fit X-learner stage-one and stage-two models.\n\n        Args:\n+            data: Input DataFrame.\n+            covariates: Covariate column names.\n+            treatment_col: Binary treatment column.\n+            outcome_col: Outcome column.\n\n        Returns:\n+            Self.\n        """
        covariate_list, x_matrix, treatment, outcome = _prepare_meta_learning_data(
            data,
            covariates,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
        )
        _validate_group_overlap(treatment, method="XMetaLearner")

        treated_mask = treatment == 1
        control_mask = treatment == 0
        self._covariates = covariate_list
        self.outcome_model_t.fit(
            np.column_stack([x_matrix[treated_mask], np.ones(np.sum(treated_mask), dtype=float)]),
            outcome[treated_mask],
        )
        self.outcome_model_c.fit(
            np.column_stack([x_matrix[control_mask], np.zeros(np.sum(control_mask), dtype=float)]),
            outcome[control_mask],
        )

        y1_hat = self.outcome_model_t.predict(np.column_stack([x_matrix, np.ones(len(x_matrix), dtype=float)]))
        y0_hat = self.outcome_model_c.predict(np.column_stack([x_matrix, np.zeros(len(x_matrix), dtype=float)]))
        imputed_treated_effect = outcome - y0_hat
        imputed_control_effect = y1_hat - outcome

        self.effect_model_t.fit(
            np.column_stack([x_matrix[treated_mask], np.ones(np.sum(treated_mask), dtype=float)]),
            imputed_treated_effect[treated_mask],
        )
        self.effect_model_c.fit(
            np.column_stack([x_matrix[control_mask], np.ones(np.sum(control_mask), dtype=float)]),
            imputed_control_effect[control_mask],
        )
        self.ate_ = float(np.mean(imputed_treated_effect))
        self._is_fitted = True
        return self

    def predict_cate(self, x: pd.DataFrame) -> np.ndarray:
        """Predict CATE by averaging treated and control effect models."""

        self._require_fitted()
        if self.ate_ is None:
            raise ValueError("Model is not fitted.")

        x_matrix = _validate_prediction_frame(x, self._covariates, name="x")
        x_matrix = np.column_stack([x_matrix, np.ones(len(x_matrix), dtype=float)])
        return self.effect_model_t.predict(x_matrix) * 0.5 + self.effect_model_c.predict(x_matrix) * 0.5
