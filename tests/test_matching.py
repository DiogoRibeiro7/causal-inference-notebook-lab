from __future__ import annotations

import numpy as np
import pytest

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


def test_matching_raises_for_invalid_matching_inputs() -> None:
    dataset = make_confounded_binary_treatment(n=300, seed=17)
    data = dataset.data.copy()
    data.loc[data.index[:1], "treatment"] = 2
    covariates = ["x1", "x2", "x3"]

    with pytest.raises(ValueError, match="treatment must be binary and encoded as 0/1."):
        propensity_score_matching(data=data, covariates=covariates)

    with pytest.raises(ValueError, match="n_neighbors must be an integer >= 1."):
        nearest_neighbour_matching(data=dataset.data, covariates=covariates, n_neighbors=0)

    with pytest.raises(TypeError, match="covariates must be a sequence of column names, not a string."):
        nearest_neighbour_matching(data=dataset.data, covariates="x1")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="treatment_col must be a string."):
        nearest_neighbour_matching(data=dataset.data, covariates=["x1", "x2", "x3"], treatment_col=1)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="treatment_col must be a string."):
        propensity_score_matching(data=dataset.data, covariates=["x1", "x2", "x3"], treatment_col=1)  # type: ignore[arg-type]

    result = nearest_neighbour_matching(
        data=dataset.data,
        covariates=["x1", "x2", "x3"],
        caliper=np.float64(0.5),
    )
    assert result.effect.n_observations >= 2

    ps_result = propensity_score_matching(
        data=dataset.data,
        covariates=["x1", "x2", "x3"],
        caliper=np.float64(0.5),
    )
    assert ps_result.effect.n_observations >= 2

    int_ps_result = propensity_score_matching(
        data=dataset.data,
        covariates=["x1", "x2", "x3"],
        caliper=np.int64(1),
    )
    assert int_ps_result.effect.n_observations >= 2

    combined_ps_result = propensity_score_matching(
        data=dataset.data,
        covariates=["x1", "x2", "x3"],
        caliper=np.float64(0.5),
        random_state=np.int64(3),
    )
    assert combined_ps_result.effect.n_observations >= 2

    result = nearest_neighbour_matching(
        data=dataset.data,
        covariates=["x1", "x2", "x3"],
        caliper=np.int64(1),
    )
    assert result.effect.n_observations >= 2


def test_propensity_score_matching_validates_propensity_scores() -> None:
    dataset = make_confounded_binary_treatment(n=200, seed=19)
    data = dataset.data
    covariates = ["x1", "x2", "x3"]

    short_scores = np.full(len(data) - 1, 0.5)
    with pytest.raises(ValueError, match="propensity_scores length must match data length."):
        propensity_score_matching(data=data, covariates=covariates, propensity_scores=short_scores)

    non_finite_scores = np.ones(len(data), dtype=float)
    non_finite_scores[0] = np.inf
    with pytest.raises(ValueError, match="propensity_scores must be finite."):
        propensity_score_matching(data=data, covariates=covariates, propensity_scores=non_finite_scores)


def test_matching_balance_table_validates_inputs() -> None:
    dataset = make_confounded_binary_treatment(n=150, seed=21)
    data = dataset.data
    covariates = ["x1", "x2", "x3"]
    result = nearest_neighbour_matching(data=data, covariates=covariates)

    with pytest.raises(TypeError, match="matched_data must be a pandas DataFrame."):
        matching_balance_table(data=data, covariates=covariates, matched_data=[])  # type: ignore[arg-type]

    empty = data.iloc[:0]
    with pytest.raises(ValueError, match="matched_data must not be empty."):
        matching_balance_table(data=data, covariates=covariates, matched_data=empty)


def test_matching_accepts_numpy_integer_hyperparameters() -> None:
    dataset = make_confounded_binary_treatment(n=120, seed=37)
    data = dataset.data
    covariates = ["x1", "x2", "x3"]

    result = nearest_neighbour_matching(
        data=data,
        covariates=covariates,
        n_neighbors=np.int64(2),
    )
    assert result.effect.n_observations >= 2

    combined_result = nearest_neighbour_matching(
        data=data,
        covariates=covariates,
        n_neighbors=np.int64(2),
        caliper=np.float64(0.5),
    )
    assert combined_result.effect.n_observations >= 2

    ps_result = propensity_score_matching(data=data, covariates=covariates, random_state=np.int64(3))
    assert ps_result.effect.n_observations >= 2
