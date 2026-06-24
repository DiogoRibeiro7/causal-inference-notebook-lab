from __future__ import annotations

import numpy as np
import pytest

from causal_inference_lab.data_generators import make_synthetic_control_data
from causal_inference_lab.synthetic_control import fit_synthetic_control


def test_synthetic_control_reproduces_known_effect() -> None:
    dataset = make_synthetic_control_data(
        n_units=20,
        n_periods=12,
        pre_periods=6,
        effect=3.0,
        seed=22,
    )
    data = dataset.data
    result = fit_synthetic_control(data, treated_unit=0, pre_period_end=5)

    assert abs(result.estimated_effect - dataset.true_ate) < 1.0
    assert abs(result.weights["weight"].sum() - 1.0) < 1e-6
    assert (result.weights["weight"] >= 0).all()


def test_synthetic_control_input_validation() -> None:
    dataset = make_synthetic_control_data(
        n_units=6,
        n_periods=8,
        pre_periods=4,
        effect=2.0,
        seed=11,
    )
    data = dataset.data

    with pytest.raises(TypeError, match="data must be a pandas DataFrame."):
        fit_synthetic_control(data="not-a-dataframe", treated_unit=0)  # type: ignore[arg-type]

    with pytest.raises(
        TypeError,
        match="unit_col, time_col, outcome_col, and treatment_col must be strings.",
    ):
        fit_synthetic_control(data, treated_unit=0, unit_col=1)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="treated_unit must be an int or a string."):
        fit_synthetic_control(data, treated_unit=1.2)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="treated_unit must be an int or a string."):
        fit_synthetic_control(data, treated_unit=True)  # type: ignore[arg-type]

    result = fit_synthetic_control(data, treated_unit=np.int64(0), pre_period_end=3)
    assert abs(result.estimated_effect - dataset.true_ate) < 1.0

    with pytest.raises(ValueError, match="Missing required columns"):
        fit_synthetic_control(data.drop(columns=["time"]), treated_unit=0)

    with pytest.raises(ValueError, match="required columns must not contain missing values"):
        bad = data.copy()
        bad.loc[bad.index[0], "outcome"] = np.nan
        fit_synthetic_control(bad, treated_unit=0, pre_period_end=3)

    with pytest.raises(ValueError, match="treatment_col must be binary and encoded as 0/1."):
        nonbinary = data.copy()
        nonbinary.loc[nonbinary.index[0], "treatment"] = 2
        fit_synthetic_control(nonbinary, treated_unit=0, pre_period_end=3)

    with pytest.raises(TypeError, match="pre_period_end must be an integer."):
        fit_synthetic_control(data, treated_unit=0, pre_period_end=3.5)

    with pytest.raises(TypeError, match="pre_period_end must be an integer."):
        fit_synthetic_control(data, treated_unit=0, pre_period_end=True)

    result = fit_synthetic_control(data, treated_unit=0, pre_period_end=np.int64(3))
    assert abs(result.estimated_effect - dataset.true_ate) < 1.0

    combined_numpy_result = fit_synthetic_control(
        data,
        treated_unit=np.int64(0),
        pre_period_end=np.int64(3),
    )
    assert abs(combined_numpy_result.estimated_effect - dataset.true_ate) < 1.0

    with pytest.raises(
        ValueError,
        match="pre_period_end must be within observed treatment periods.",
    ):
        fit_synthetic_control(data, treated_unit=0, pre_period_end=20)
