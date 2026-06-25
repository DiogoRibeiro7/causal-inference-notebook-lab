from __future__ import annotations

import numpy as np
import pytest

from causal_inference_lab.data_generators import make_confounded_binary_treatment
from causal_inference_lab.diagnostics import (
    balance_table,
    ipw_weights,
    overlap_summary,
    standardized_mean_difference,
)


def test_diagnostics_standardized_mean_difference_input_validation() -> None:
    dataset = make_confounded_binary_treatment(n=40, seed=13)
    data = dataset.data

    with pytest.raises(TypeError, match="data must be a pandas DataFrame."):
        standardized_mean_difference("bad", covariate="x1")

    with pytest.raises(TypeError, match="covariate must be a string."):
        standardized_mean_difference(data, covariate=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown covariate"):
        standardized_mean_difference(data, covariate="missing")

    bad = data.copy()
    bad.loc[bad.index[0], "x1"] = np.nan
    with pytest.raises(ValueError, match="covariate values must be finite."):
        standardized_mean_difference(bad, covariate="x1")

    weights = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="weights must have the same length as data."):
        standardized_mean_difference(data, covariate="x1", weights=weights)


def test_diagnostics_balance_table_input_validation() -> None:
    data = make_confounded_binary_treatment(n=50, seed=21).data

    with pytest.raises(TypeError, match="data must be a pandas DataFrame."):
        balance_table("bad", covariates=["x1", "x2"])  # type: ignore[arg-type]

    with pytest.raises(
        TypeError, match="covariates must be a sequence of column names, not a string."
    ):
        balance_table(data, covariates="x1")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="covariates must not be empty."):
        balance_table(data, covariates=[])


def test_diagnostics_ipw_weights_input_validation() -> None:
    data = make_confounded_binary_treatment(n=80, seed=5).data.copy()

    with pytest.raises(
        TypeError, match="covariates must be a sequence of column names, not a string."
    ):
        ipw_weights(data, covariates="x1")  # type: ignore[arg-type]

    weights = ipw_weights(data, covariates=["x1", "x2", "x3"], clip=np.float64(0.02))
    assert np.isfinite(weights).all()

    data.loc[data.index[:1], "treatment"] = 2
    with pytest.raises(ValueError, match="treatment must be binary and encoded as 0/1."):
        ipw_weights(data, covariates=["x1", "x2", "x3"])


def test_overlap_summary_input_validation() -> None:
    with pytest.raises(ValueError, match="propensity_scores must be one-dimensional."):
        overlap_summary(np.array([[0.1, 0.2]]))

    with pytest.raises(ValueError, match="propensity_scores must not be empty."):
        overlap_summary(np.array([]))

    with pytest.raises(ValueError, match="propensity_scores must be finite."):
        overlap_summary(np.array([0.1, np.nan]))
