from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_inference_lab.data_generators import make_confounded_binary_treatment
from causal_inference_lab.estimators import ipw_ate
from causal_inference_lab.sensitivity import omitted_confounder_simulation, placebo_treatment_test


def test_placebo_treatment_test_validates_inputs() -> None:
    dataset = make_confounded_binary_treatment(n=300, seed=11)
    data = dataset.data

    with pytest.raises(TypeError, match="data must be a pandas DataFrame."):
        placebo_treatment_test(
            data="not-a-dataframe",  # type: ignore[arg-type]
            estimator=ipw_ate,
            covariates=["x1", "x2", "x3"],
        )

    with pytest.raises(TypeError, match="estimator must be callable."):
        placebo_treatment_test(
            data=data,
            estimator="not-callable",  # type: ignore[arg-type]
            covariates=["x1", "x2", "x3"],
        )

    with pytest.raises(
        TypeError, match="covariates must be a sequence of column names, not a string."
    ):
        placebo_treatment_test(
            data=data,
            estimator=ipw_ate,
            covariates="x1",  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="estimator must return an EffectEstimate."):
        placebo_treatment_test(
            data=data,
            estimator=lambda _frame, _covariates: 0.5,
            covariates=["x1", "x2", "x3"],
        )

    with pytest.raises(TypeError, match="seed must be an integer."):
        placebo_treatment_test(
            data=data,
            estimator=ipw_ate,
            covariates=["x1", "x2", "x3"],
            seed=1.5,
        )

    placebo_result = placebo_treatment_test(
        data=data,
        estimator=ipw_ate,
        covariates=["x1", "x2", "x3"],
        seed=np.int64(8),
    )
    assert np.isfinite(placebo_result.estimate)


def test_placebo_treatment_test_rejects_missing_treatment_or_invalid_treatment_data() -> None:
    dataset = make_confounded_binary_treatment(n=200, seed=7)
    no_treatment = dataset.data.drop(columns=["treatment"])

    with pytest.raises(ValueError, match="Unknown treatment column: treatment"):
        placebo_treatment_test(
            data=no_treatment,
            estimator=ipw_ate,
            covariates=["x1", "x2", "x3"],
        )

    with pytest.raises(TypeError, match="treatment_col must be a string."):
        placebo_treatment_test(
            data=dataset.data,
            estimator=ipw_ate,
            covariates=["x1", "x2", "x3"],
            treatment_col=1,  # type: ignore[arg-type]
        )

    bad_treatment = dataset.data.copy()
    bad_treatment.loc[bad_treatment.index[:2], "treatment"] = 2
    with pytest.raises(ValueError, match="treatment must be binary and encoded as 0/1."):
        placebo_treatment_test(
            data=bad_treatment,
            estimator=ipw_ate,
            covariates=["x1", "x2", "x3"],
        )


def test_omitted_confounder_simulation_input_validation() -> None:
    dataset = make_confounded_binary_treatment(n=200, seed=17)
    data = dataset.data

    with pytest.raises(TypeError, match="base_effect must be a numeric value."):
        omitted_confounder_simulation(
            data=data,
            base_effect="0.5",  # type: ignore[arg-type]
            confounder_strength_grid=[0.0, 0.1],
        )

    with pytest.raises(ValueError, match="base_effect must be finite."):
        omitted_confounder_simulation(
            data=data,
            base_effect=np.inf,
            confounder_strength_grid=[0.0, 0.1],
        )

    with pytest.raises(ValueError, match="confounder_strength_grid must not be empty."):
        omitted_confounder_simulation(
            data=data,
            base_effect=dataset.true_ate,
            confounder_strength_grid=[],
        )

    with pytest.raises(ValueError, match="confounder_strength_grid must contain numeric values."):
        omitted_confounder_simulation(
            data=data,
            base_effect=dataset.true_ate,
            confounder_strength_grid=["bad"],
        )

    with pytest.raises(ValueError, match="confounder_strength_grid values must be non-negative."):
        omitted_confounder_simulation(
            data=data,
            base_effect=dataset.true_ate,
            confounder_strength_grid=[-0.2],
        )

    with pytest.raises(ValueError, match="confounder_strength_grid values must be finite."):
        omitted_confounder_simulation(
            data=data,
            base_effect=dataset.true_ate,
            confounder_strength_grid=[0.1, float("inf")],
        )

    with pytest.raises(TypeError, match="seed must be an integer."):
        omitted_confounder_simulation(
            data=data,
            base_effect=dataset.true_ate,
            confounder_strength_grid=[0.0],
            seed=0.5,
        )

    simulation = omitted_confounder_simulation(
        data=data,
        base_effect=np.float64(dataset.true_ate),
        confounder_strength_grid=[0.0],
        seed=np.int64(3),
    )
    assert not simulation.empty

    numpy_simulation = omitted_confounder_simulation(
        data=data,
        base_effect=np.float64(dataset.true_ate),
        confounder_strength_grid=[np.float64(0.0), np.float64(0.1)],
        seed=np.int64(7),
    )
    assert numpy_simulation.shape[0] == 2

    integer_strength_simulation = omitted_confounder_simulation(
        data=data,
        base_effect=np.float64(dataset.true_ate),
        confounder_strength_grid=[np.int64(0), np.int64(1)],
        seed=np.int64(9),
    )
    assert integer_strength_simulation.shape[0] == 2

    integer_base_effect_simulation = omitted_confounder_simulation(
        data=data,
        base_effect=np.int64(1),
        confounder_strength_grid=[np.float64(0.0), np.float64(0.1)],
        seed=np.int64(11),
    )
    assert integer_base_effect_simulation.shape[0] == 2


def test_omitted_confounder_simulation_rejects_no_treatment_variation() -> None:
    data = pd.DataFrame(
        {
            "treatment": [1, 1, 1, 1, 1],
            "x1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "outcome": [1.0, 1.2, 1.1, 1.4, 1.0],
        }
    )

    with pytest.raises(ValueError, match="Treatment must have variation for sensitivity analysis."):
        omitted_confounder_simulation(
            data=data,
            base_effect=1.0,
            confounder_strength_grid=[0.0, 0.2],
        )


def test_omitted_confounder_simulation_rejects_nonfinite_treatment_values() -> None:
    dataset = make_confounded_binary_treatment(n=200, seed=17)

    data = pd.DataFrame(
        {
            "treatment": [0, 1, 0, np.nan, 1],
            "x1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "outcome": [1.0, 1.2, 1.1, 1.4, 1.0],
        }
    )

    with pytest.raises(ValueError, match="Treatment values must be finite."):
        omitted_confounder_simulation(
            data=data,
            base_effect=0.8,
            confounder_strength_grid=[0.0, 0.2],
        )

    bad_treatment = pd.DataFrame(
        {
            "treatment": [0, 1, 2, 1, 0],
            "x1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "outcome": [1.0, 1.2, 1.1, 1.4, 1.0],
        }
    )
    with pytest.raises(ValueError, match="treatment must be binary and encoded as 0/1."):
        omitted_confounder_simulation(data=bad_treatment, base_effect=dataset.true_ate, confounder_strength_grid=[0.0])
