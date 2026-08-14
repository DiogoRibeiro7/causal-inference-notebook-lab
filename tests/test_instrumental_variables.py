from __future__ import annotations

import numpy as np
import pytest

from causal_inference_lab.data_generators import make_iv_data
from causal_inference_lab.estimators import ols_treatment_effect
from causal_inference_lab.instrumental_variables import instrumental_variables_ate


def test_instrumental_variables_recover_ate_better_than_ols() -> None:
    dataset = make_iv_data(n=6_000, seed=11)
    data = dataset.data

    iv_result = instrumental_variables_ate(data)
    # The OLS baseline may only adjust for the observed covariate; the confounder is
    # unobserved, so OLS stays biased and IV should recover the effect more accurately.
    ols_result = ols_treatment_effect(
        data,
        covariates=["x"],
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


def test_instrumental_variables_rejects_invalid_inputs() -> None:
    dataset = make_iv_data(n=300, seed=20)
    data = dataset.data

    with pytest.raises(
        TypeError, match="covariates must be a sequence of column names, not a string."
    ):
        instrumental_variables_ate(data, covariates="x")

    with pytest.raises(ValueError, match="covariates must be unique."):
        instrumental_variables_ate(data, covariates=["x", "x"])

    with pytest.raises(TypeError, match="covariates must be a sequence of strings."):
        instrumental_variables_ate(data, covariates=[["x"]])  # type: ignore[list-item]

    bad_treatment = data.copy()
    bad_treatment.loc[bad_treatment.index[:1], "treatment"] = 2
    with pytest.raises(ValueError, match="treatment must be binary and encoded as 0/1."):
        instrumental_variables_ate(bad_treatment)

    with pytest.raises(TypeError, match="treatment_col must be a string."):
        instrumental_variables_ate(data, treatment_col=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="treatment_col and instrument_col must be different."):
        instrumental_variables_ate(data, treatment_col="instrument", instrument_col="instrument")


def test_iv_reports_late_not_ate() -> None:
    """2SLS identifies the complier effect, and the result object must say so.

    Labelling it "ATE" invites the exact misreading the assumptions matrix warns
    about: the two coincide only under homogeneous effects, which is the
    assumption an instrument is usually invoked to avoid.
    """
    dataset = make_iv_data(n=500, seed=99)
    result = instrumental_variables_ate(dataset.data)
    assert result.effect.estimand == "LATE"
