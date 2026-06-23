from __future__ import annotations

import numpy as np

from causal_inference_lab.data_generators import make_iv_data
from causal_inference_lab.estimators import ols_treatment_effect
from causal_inference_lab.instrumental_variables import instrumental_variables_ate


def test_instrumental_variables_recover_ate_better_than_ols() -> None:
    dataset = make_iv_data(n=6_000, seed=11)
    data = dataset.data

    iv_result = instrumental_variables_ate(data)
    ols_result = ols_treatment_effect(
        data,
        covariates=["x", "hidden_confounder"],
        treatment_col="treatment",
        outcome_col="outcome",
    )

    assert iv_result.effect.estimator == "instrumental_variables_2sls"
    assert iv_result.effect.n_observations == len(data)
    assert abs(iv_result.effect.estimate - dataset.true_ate) < 0.8
    assert iv_result.first_stage_f_stat > 0.0
    assert iv_result.first_stage_f_stat < 10_000
    assert np.isfinite(iv_result.first_stage_r2)
    iv_bias = abs(iv_result.effect.estimate - dataset.true_ate)
    ols_bias = abs(ols_result.estimate - dataset.true_ate)
    assert iv_bias < ols_bias
    assert isinstance(iv_result.instrument_is_weak, bool)


def test_instrumental_variables_rejects_constant_instrument() -> None:
    dataset = make_iv_data(n=300, seed=20)
    bad = dataset.data.copy()
    bad["instrument"] = 1

    try:
        instrumental_variables_ate(bad)
    except ValueError as exc:
        assert "Instrument must vary" in str(exc)
    else:
        raise AssertionError("Expected ValueError for constant instrument.")
