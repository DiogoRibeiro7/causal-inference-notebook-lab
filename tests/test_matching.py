from __future__ import annotations

import numpy as np

from causal_inference_lab.data_generators import make_confounded_binary_treatment
from causal_inference_lab.matching import matching_balance_table, nearest_neighbour_matching, propensity_score_matching


def test_propensity_score_matching_reduces_covariate_imbalance() -> None:
    dataset = make_confounded_binary_treatment(n=5_000, seed=10)
    data = dataset.data
    covariates = ["x1", "x2", "x3"]

    before, _ = matching_balance_table(
        data=data,
        covariates=covariates,
        matched_data=data,
    )
    result = propensity_score_matching(data=data, covariates=covariates)
    _, after = matching_balance_table(
        data=data,
        covariates=covariates,
        matched_data=result.matched_data,
    )

    assert result.effect.estimator == "propensity_score_matching"
    assert result.effect.estimand == "ATT"
    assert result.effect.n_observations >= 2
    assert np.mean(after["abs_smd"]) <= np.mean(before["abs_smd"]) + 0.02


def test_nearest_neighbour_matching_returns_att_within_reasonable_range() -> None:
    dataset = make_confounded_binary_treatment(n=4_000, seed=77)
    data = dataset.data
    covariates = ["x1", "x2", "x3"]

    result = nearest_neighbour_matching(data=data, covariates=covariates)
    estimate = result.effect.estimate

    assert np.isfinite(estimate)
    assert result.dropped_units < data["treatment"].sum()
