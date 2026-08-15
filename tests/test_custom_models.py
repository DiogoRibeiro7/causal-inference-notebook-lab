"""Every estimator that accepts a caller-supplied model must accept an ensemble.

These entry points all defaulted their model with ``model or Default()``. That
looks harmless, but ``or`` evaluates ``bool(model)``, and an unfitted scikit-learn
ensemble raises ``AttributeError`` from ``__len__`` because ``estimators_`` does
not exist until ``fit``. So passing a gradient-boosted or random-forest nuisance
model crashed before doing any work.

The failure hit exactly the use case these parameters exist for: double machine
learning and meta-learners are worth reaching for precisely when the nuisance
functions are non-linear, which is when you supply an ensemble.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from causal_inference_lab.data_generators import make_confounded_binary_treatment
from causal_inference_lab.dml import double_machine_learning_ate
from causal_inference_lab.estimators import g_computation_ate
from causal_inference_lab.meta_learners import SMetaLearner, TMetaLearner, XMetaLearner

COVARIATES = ["x1", "x2", "x3"]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return make_confounded_binary_treatment(n=600, seed=5).data


def _forest() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=8, random_state=0)


def test_g_computation_accepts_an_ensemble_outcome_model(data: pd.DataFrame) -> None:
    result = g_computation_ate(data, covariates=COVARIATES, outcome_model=_forest())
    assert result.estimand == "ATE"


def test_dml_accepts_an_ensemble_outcome_model(data: pd.DataFrame) -> None:
    result = double_machine_learning_ate(
        data,
        covariates=COVARIATES,
        n_splits=2,
        seed=0,
        outcome_model=GradientBoostingRegressor(n_estimators=10, random_state=0),
    )
    assert result.estimand == "ATE"


def test_dml_accepts_an_ensemble_treatment_model(data: pd.DataFrame) -> None:
    result = double_machine_learning_ate(
        data,
        covariates=COVARIATES,
        n_splits=2,
        seed=0,
        treatment_model=RandomForestClassifier(n_estimators=8, random_state=0),
    )
    assert result.estimand == "ATE"


@pytest.mark.parametrize(
    "learner",
    [
        SMetaLearner(model=_forest()),
        TMetaLearner(treated_model=_forest(), control_model=_forest()),
        XMetaLearner(
            outcome_model_t=_forest(),
            outcome_model_c=_forest(),
            effect_model_t=_forest(),
            effect_model_c=_forest(),
        ),
    ],
    ids=["s-learner", "t-learner", "x-learner"],
)
def test_meta_learners_accept_ensemble_models(learner: object, data: pd.DataFrame) -> None:
    fitted = learner.fit(  # type: ignore[attr-defined]
        data,
        covariates=COVARIATES,
        treatment_col="treatment",
        outcome_col="outcome",
    )
    cate = fitted.predict_cate(data[COVARIATES])
    assert len(cate) == len(data)


def test_supplied_model_is_actually_used(data: pd.DataFrame) -> None:
    """A supplied model must be fitted, not silently replaced by the default."""
    forest = _forest()
    g_computation_ate(data, covariates=COVARIATES, outcome_model=forest)
    assert hasattr(forest, "estimators_"), "the supplied model was never fitted"
