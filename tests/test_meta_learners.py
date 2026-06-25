from __future__ import annotations

import numpy as np

from causal_inference_lab.data_generators import make_heterogeneous_treatment_data
from causal_inference_lab.meta_learners import SMetaLearner, TMetaLearner, XMetaLearner


def test_meta_learners_estimate_ate_on_synthetic_data() -> None:
    dataset = make_heterogeneous_treatment_data(n=4_000, seed=10)
    data = dataset.data
    covariates = ["age", "risk_score", "prior_usage"]

    s_learner = SMetaLearner().fit(
        data, covariates, treatment_col="treatment", outcome_col="outcome"
    )
    t_learner = TMetaLearner().fit(
        data, covariates, treatment_col="treatment", outcome_col="outcome"
    )
    x_learner = XMetaLearner().fit(
        data, covariates, treatment_col="treatment", outcome_col="outcome"
    )

    s_ate = s_learner.estimate_ate(data.loc[:, covariates])
    t_ate = t_learner.estimate_ate(data.loc[:, covariates])
    x_ate = x_learner.estimate_ate(data.loc[:, covariates])

    assert np.isfinite(s_ate)
    assert np.isfinite(t_ate)
    assert np.isfinite(x_ate)

    assert len(s_learner.predict_cate(data.loc[:, covariates])) == len(data)
    assert np.allclose(
        s_learner.predict_cate(data.loc[:, covariates]),
        SMetaLearner()
        .fit(data, covariates, treatment_col="treatment", outcome_col="outcome")
        .predict_cate(data.loc[:, covariates]),
    )
    assert np.allclose(
        t_learner.predict_cate(data.loc[:, covariates]),
        TMetaLearner()
        .fit(data, covariates, treatment_col="treatment", outcome_col="outcome")
        .predict_cate(data.loc[:, covariates]),
    )

    assert abs(s_ate - dataset.true_ate) < 1.0
    assert abs(t_ate - dataset.true_ate) < 1.0


def test_meta_learners_reject_non_binary_treatment() -> None:
    dataset = make_heterogeneous_treatment_data(n=500, seed=11)
    data = dataset.data.copy()
    covariates = ["age", "risk_score", "prior_usage"]
    data.loc[data.index[:2], "treatment"] = 2

    for learner in (SMetaLearner(), TMetaLearner(), XMetaLearner()):
        try:
            learner.fit(data, covariates, treatment_col="treatment", outcome_col="outcome")
        except ValueError as err:
            assert str(err) == "treatment must be binary and encoded as 0/1."
        else:
            raise AssertionError("Expected ValueError for non-binary treatment.")


def test_meta_learners_reject_missing_columns() -> None:
    dataset = make_heterogeneous_treatment_data(n=250, seed=12)
    data = dataset.data
    covariates = ["age", "risk_score", "prior_usage", "missing_col"]

    for learner in (SMetaLearner(), TMetaLearner(), XMetaLearner()):
        try:
            learner.fit(data, covariates, treatment_col="treatment", outcome_col="outcome")
        except ValueError as err:
            assert str(err) == "data is missing required covariate columns: ['missing_col']."
        else:
            raise AssertionError("Expected ValueError for missing covariate columns.")


def test_meta_learners_require_fit_before_predict() -> None:
    dataset = make_heterogeneous_treatment_data(n=400, seed=13)
    x = dataset.data[["age", "risk_score", "prior_usage"]]

    for learner in (SMetaLearner(), TMetaLearner(), XMetaLearner()):
        try:
            learner.predict_cate(x)
        except ValueError as err:
            assert str(err) == "Model has not been fitted. Call fit(...) first."
        else:
            raise AssertionError("Expected ValueError when predicting before fit.")
