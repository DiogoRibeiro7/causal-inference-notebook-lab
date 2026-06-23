"""Heterogeneous treatment effect meta-learners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression


@dataclass
class CATEModel:
    """Container for a trained CATE estimator."""

    estimator: RegressorMixin

    def predict_cate(self, x: pd.DataFrame) -> np.ndarray:
        """Predict CATE for each row in ``x``.\n\n        Subclasses must implement this method."""
        raise NotImplementedError

    def estimate_ate(self, x: pd.DataFrame) -> float:
        """Return the average predicted CATE."""
        return float(np.mean(self.predict_cate(x)))


class SMetaLearner(CATEModel):
    """S-learner: outcome model includes treatment indicator as a feature."""

    def __init__(self, model: RegressorMixin | None = None) -> None:
        self.model = model or LinearRegression()

    def _features(self, x: pd.DataFrame, treatment: pd.Series | None = None) -> np.ndarray:
        if treatment is None:
            treatment = pd.Series(np.zeros(len(x), dtype=float), index=x.index)
        features = x.copy()
        features["treatment"] = treatment.astype(float)
        return features.to_numpy(dtype=float)

    def fit(self, data: pd.DataFrame, covariates: Sequence[str], treatment_col: str, outcome_col: str) -> "SMetaLearner":
        """Fit the S-learner outcome model.\n\n        Args:\n            data: Input dataset.\n            covariates: Covariate columns.\n            treatment_col: Binary treatment column.\n            outcome_col: Outcome column.\n\n        Returns:\n            Self.\n        """
        x = data.loc[:, covariates]
        t = data[treatment_col]
        y = data[outcome_col]
        self.model.fit(self._features(x, t), y.to_numpy(dtype=float))
        return self

    def predict_cate(self, x: pd.DataFrame) -> np.ndarray:
        treated = self._features(x, pd.Series(np.ones(len(x)), index=x.index))
        untreated = self._features(x, pd.Series(np.zeros(len(x)), index=x.index))
        return self.model.predict(treated) - self.model.predict(untreated)


class TMetaLearner(CATEModel):
    """T-learner: separate outcome models in treated and control groups."""

    def __init__(
        self,
        treated_model: RegressorMixin | None = None,
        control_model: RegressorMixin | None = None,
    ) -> None:
        self.treated_model = treated_model or LinearRegression()
        self.control_model = control_model or LinearRegression()

    def fit(self, data: pd.DataFrame, covariates: Sequence[str], treatment_col: str, outcome_col: str) -> "TMetaLearner":
        """Fit separate outcome models for treated and control rows.\n\n        Args:\n            data: Input dataset.\n            covariates: Covariate columns.\n            treatment_col: Binary treatment column.\n            outcome_col: Outcome column.\n\n        Returns:\n            Self.\n        """
        x = data.loc[:, covariates]
        t = data[treatment_col].to_numpy(dtype=int)
        y = data[outcome_col]
        treated_data = data.loc[t == 1]
        control_data = data.loc[t == 0]
        if treated_data.empty or control_data.empty:
            raise ValueError("Both treated and control groups are required.")

        self.treated_model.fit(treated_data.loc[:, covariates], treated_data[outcome_col])
        self.control_model.fit(control_data.loc[:, covariates], control_data[outcome_col])
        return self

    def predict_cate(self, x: pd.DataFrame) -> np.ndarray:
        return self.treated_model.predict(x.to_numpy(dtype=float)) - self.control_model.predict(
            x.to_numpy(dtype=float)
        )


class XMetaLearner(CATEModel):
    """X-learner with simple stage-two CATE models."""

    def __init__(
        self,
        outcome_model_t: RegressorMixin | None = None,
        outcome_model_c: RegressorMixin | None = None,
        effect_model_t: RegressorMixin | None = None,
        effect_model_c: RegressorMixin | None = None,
    ) -> None:
        self.outcome_model_t = outcome_model_t or LinearRegression()
        self.outcome_model_c = outcome_model_c or LinearRegression()
        self.effect_model_t = effect_model_t or LinearRegression()
        self.effect_model_c = effect_model_c or LinearRegression()
        self.ate_: float | None = None

    def fit(self, data: pd.DataFrame, covariates: Sequence[str], treatment_col: str, outcome_col: str) -> "XMetaLearner":
        """Fit X-learner stage-one and stage-two models.\n\n        Args:\n            data: Input dataset.\n            covariates: Covariate columns.\n            treatment_col: Binary treatment column.\n            outcome_col: Outcome column.\n\n        Returns:\n            Self.\n        """
        x = data.loc[:, covariates]
        t = data[treatment_col].to_numpy(dtype=int)
        y = data[outcome_col].to_numpy(dtype=float)
        treated = data[t == 1]
        control = data[t == 0]
        if treated.empty or control.empty:
            raise ValueError("Both treated and control groups are required.")

        if x.isnull().any().any():
            raise ValueError("x contains missing values; remove or impute before fitting.")
        if treated.empty or control.empty:
            raise ValueError("Both treated and control groups are required.")

        self.outcome_model_t.fit(
            np.column_stack([treated.loc[:, covariates], np.ones(len(treated))]),
            treated[outcome_col].to_numpy(dtype=float),
        )
        self.outcome_model_c.fit(
            np.column_stack([control.loc[:, covariates], np.zeros(len(control))]),
            control[outcome_col].to_numpy(dtype=float),
        )

        y1_hat = self.outcome_model_t.predict(np.column_stack([x.to_numpy(dtype=float), np.ones(len(x))]))
        y0_hat = self.outcome_model_c.predict(np.column_stack([x.to_numpy(dtype=float), np.zeros(len(x))]))
        uplift = y - y0_hat
        uplift_c = y1_hat - y

        treated_x = x.loc[data[treatment_col] == 1]
        control_x = x.loc[data[treatment_col] == 0]
        effect_model_t_features = np.column_stack(
            [treated_x.to_numpy(dtype=float), np.ones(len(treated_x))],
        )
        effect_model_c_features = np.column_stack(
            [control_x.to_numpy(dtype=float), np.ones(len(control_x))],
        )
        self.effect_model_t.fit(effect_model_t_features, uplift[data[treatment_col] == 1].to_numpy(dtype=float))
        self.effect_model_c.fit(effect_model_c_features, uplift_c[data[treatment_col] == 0].to_numpy(dtype=float))
        self.ate_ = float(np.mean(uplift))
        return self

    def predict_cate(self, x: pd.DataFrame) -> np.ndarray:
        if self.ate_ is None:
            raise ValueError("Model is not fitted.")

        x_matrix = np.column_stack([x.to_numpy(dtype=float), np.ones(len(x))])
        return self.effect_model_t.predict(x_matrix) * 0.5 + self.effect_model_c.predict(x_matrix) * 0.5
