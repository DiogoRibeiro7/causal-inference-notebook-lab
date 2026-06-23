from __future__ import annotations

import numpy as np

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
