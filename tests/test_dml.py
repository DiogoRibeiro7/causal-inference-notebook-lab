from __future__ import annotations

import numpy as np
import pytest

from causal_inference_lab.data_generators import make_confounded_binary_treatment
from causal_inference_lab.dml import double_machine_learning_ate
from causal_inference_lab.estimators import ols_treatment_effect


def test_dml_estimate_close_to_true_ate_on_synthetic_data() -> None:
    dataset = make_confounded_binary_treatment(n=5_000, seed=7)
    data = dataset.data
    covariates = ["x1", "x2", "x3"]

    dml_effect = double_machine_learning_ate(data, covariates)
    ols_effect = ols_treatment_effect(data, covariates)

    assert np.isfinite(dml_effect.estimate)
    assert np.isfinite(ols_effect.estimate)
    assert abs(dml_effect.estimate - dataset.true_ate) < 0.6
    assert dml_effect.estimand == "ATE"


def test_dml_raises_for_invalid_inputs() -> None:
    dataset = make_confounded_binary_treatment(n=500, seed=7)
    data = dataset.data
    data.loc[data.index[:1], "treatment"] = 2
    covariates = ["x1", "x2", "x3"]

    with pytest.raises(ValueError, match="treatment must be binary and encoded as 0/1."):
        double_machine_learning_ate(data, covariates)

    with pytest.raises(ValueError, match="n_splits must be an integer >= 2."):
        double_machine_learning_ate(dataset.data, covariates, n_splits=1)

    with pytest.raises(ValueError, match="seed must be an integer."):
        double_machine_learning_ate(dataset.data, covariates, seed="123")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="n_splits cannot exceed the number of observations."):
        double_machine_learning_ate(dataset.data, covariates, n_splits=10_000)

    with pytest.raises(ValueError, match="covariates must not be empty."):
        double_machine_learning_ate(dataset.data, [])

    result = double_machine_learning_ate(
        dataset.data,
        covariates,
        n_splits=np.int64(5),
        seed=np.int64(7),
    )
    assert np.isfinite(result.estimate)
