from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_inference_lab.data_generators import make_confounded_binary_treatment
from causal_inference_lab.diagnostics import (
    balance_table,
    ipw_weights,
    standardized_mean_difference,
)
from causal_inference_lab.estimators import (
    estimate_propensity_scores,
    aipw_ate,
    difference_in_means,
    fit_propensity_model,
    g_computation_ate,
    ipw_ate,
)
from causal_inference_lab.sensitivity import omitted_confounder_simulation


def test_estimators_are_close_to_true_ate_on_synthetic_data() -> None:
    dataset = make_confounded_binary_treatment(n=4_000, seed=42)
    data = dataset.data
    covariates = ["x1", "x2", "x3"]

    naive = difference_in_means(data)
    ipw = ipw_ate(data, covariates)
    gcomp = g_computation_ate(data, covariates)
    aipw = aipw_ate(data, covariates)

    naive_error = abs(naive.estimate - dataset.true_ate)
    ipw_error = abs(ipw.estimate - dataset.true_ate)
    gcomp_error = abs(gcomp.estimate - dataset.true_ate)
    aipw_error = abs(aipw.estimate - dataset.true_ate)

    assert ipw_error < naive_error
    assert gcomp_error < naive_error
    assert aipw_error < naive_error
    assert aipw_error < 0.25


def test_ipw_weights_improve_average_balance() -> None:
    dataset = make_confounded_binary_treatment(n=4_000, seed=123)
    data = dataset.data
    covariates = ["x1", "x2", "x3"]

    before = balance_table(data, covariates)
    weights = ipw_weights(data, covariates)
    after = balance_table(data, covariates, weights=weights)

    assert np.mean(after["abs_smd"]) < np.mean(before["abs_smd"])


def test_standardized_mean_difference_uses_weighted_dispersion() -> None:
    data = pd.DataFrame(
        {
            "treatment": [1, 1, 0, 0],
            "x": [0.0, 1.0, 0.0, 10.0],
        }
    )
    weights = np.array([10.0, 1.0, 1.0, 10.0], dtype=float)

    observed = standardized_mean_difference(data, covariate="x", treatment_col="treatment", weights=weights)
    treated = data.loc[data["treatment"] == 1, "x"].to_numpy(dtype=float)
    control = data.loc[data["treatment"] == 0, "x"].to_numpy(dtype=float)
    treated_w = weights[data["treatment"] == 1]
    control_w = weights[data["treatment"] == 0]
    weighted_treated = float(np.average(treated, weights=treated_w))
    weighted_control = float(np.average(control, weights=control_w))
    treated_var = float(np.average((treated - weighted_treated) ** 2, weights=treated_w))
    control_var = float(np.average((control - weighted_control) ** 2, weights=control_w))
    expected = (weighted_treated - weighted_control) / float(np.sqrt((treated_var + control_var) / 2.0))

    assert np.isclose(observed, expected)


def test_standardized_mean_difference_rejects_single_treatment_group() -> None:
    data = pd.DataFrame({"treatment": [1, 1, 1], "x": [0.0, 1.0, 2.0]})
    try:
        _ = standardized_mean_difference(data, covariate="x", treatment_col="treatment")
    except ValueError as err:
        assert str(err) == "Both treated and control groups must contain observations."
    else:
        raise AssertionError("Expected ValueError for missing control group.")


def test_omitted_confounder_simulation_strength_increases_bias() -> None:
    dataset = make_confounded_binary_treatment(n=3_000, seed=77)
    data = dataset.data

    table = omitted_confounder_simulation(
        data,
        base_effect=dataset.true_ate,
        confounder_strength_grid=[0.0, 0.2, 0.4, 0.6],
        seed=42,
    )

    assert table.shape[0] == 4
    assert np.all(np.diff(table["simulated_bias"]) >= 0.0)
    assert table.loc[table["confounder_strength"] == 0.0, "adjusted_effect"].iloc[0] > table.loc[
        table["confounder_strength"] == 0.6,
        "adjusted_effect",
    ].iloc[0]


def test_omitted_confounder_simulation_validates_input_grid() -> None:
    dataset = make_confounded_binary_treatment(n=1_000, seed=101)
    data = dataset.data

    try:
        omitted_confounder_simulation(data, base_effect=dataset.true_ate, confounder_strength_grid=[])
    except ValueError as err:
        assert str(err) == "confounder_strength_grid must not be empty."
    else:
        raise AssertionError("Expected ValueError for empty strength grid.")

    try:
        omitted_confounder_simulation(data, base_effect=dataset.true_ate, confounder_strength_grid=[-0.1])
    except ValueError as err:
        assert str(err) == "confounder_strength_grid values must be non-negative."
    else:
        raise AssertionError("Expected ValueError for negative strength.")


def test_estimators_raises_for_invalid_treatment_and_outcome_inputs() -> None:
    dataset = make_confounded_binary_treatment(n=300, seed=7)
    data = dataset.data.copy()

    data.loc[data.index[:1], "treatment"] = 2
    with pytest.raises(ValueError, match="Treatment must be binary and encoded as 0/1."):
        difference_in_means(data, "treatment", "outcome")

    data.loc[data.index[0], "outcome"] = np.nan
    with pytest.raises(ValueError, match="outcome must be numeric and finite."):
        ipw_ate(data, ["x1", "x2", "x3"], outcome_col="outcome")

    with pytest.raises(TypeError, match="data must be a pandas DataFrame."):
        ipw_ate("not-a-dataframe", ["x1", "x2", "x3"])  # type: ignore[arg-type]


def test_estimator_helpers_validate_numeric_parameters() -> None:
    dataset = make_confounded_binary_treatment(n=200, seed=9)
    data = dataset.data

    with pytest.raises(TypeError, match="clip must be a numeric value."):
        ipw_ate(data, ["x1", "x2", "x3"], clip="0.01")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="clip must be between 0 and 0.5."):
        estimate_propensity_scores(data, ["x1", "x2", "x3"], clip=1.0)

    with pytest.raises(ValueError, match="columns must be a sequence of column names, not a string."):
        fit_propensity_model(data, covariates="x1")  # type: ignore[arg-type]
