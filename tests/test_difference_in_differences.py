from __future__ import annotations

import numpy as np
import pytest

from causal_inference_lab.data_generators import make_did_panel
from causal_inference_lab.difference_in_differences import difference_in_differences


def test_difference_in_differences_estimate_is_close_to_truth() -> None:
    dataset = make_did_panel(n_units=800, n_periods=10, seed=13)
    data = dataset.data

    result = difference_in_differences(data)

    assert result.effect.estimator == "difference_in_differences"
    assert result.effect.estimand == "ATT"
    assert abs(result.effect.estimate - dataset.true_ate) < 0.7
    assert result.effect.n_observations == len(data)
    assert result.pre_treated_mean < result.post_treated_mean
    assert result.pre_control_mean < result.post_control_mean
    assert 0.0 <= result.pre_trend_p_value <= 1.0


def test_difference_in_differences_validates_bad_grouping() -> None:
    dataset = make_did_panel(n_units=50, n_periods=6, seed=2)
    bad = dataset.data.copy()
    bad["treated_group"] = 1

    try:
        difference_in_differences(bad)
    except ValueError as exc:
        assert "exactly 2 groups" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid treated/control setup.")


def test_difference_in_differences_input_validation() -> None:
    dataset = make_did_panel(n_units=40, n_periods=6, seed=4)
    data = dataset.data

    with pytest.raises(TypeError, match="data must be a pandas DataFrame."):
        difference_in_differences(data="not-a-dataframe")  # type: ignore[arg-type]

    with pytest.raises(
        TypeError,
        match="group_col, post_col, time_col, and outcome_col must be strings.",
    ):
        difference_in_differences(data, group_col=1)  # type: ignore[arg-type]

    with pytest.raises(
        TypeError,
        match="group_col, post_col, time_col, and outcome_col must be strings.",
    ):
        difference_in_differences(data, outcome_col=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Missing required columns"):
        missing = data.drop(columns=["time"])
        difference_in_differences(missing)

    with pytest.raises(ValueError, match="required columns must not contain missing values"):
        bad = data.copy()
        bad.loc[bad.index[0], "outcome"] = np.nan
        difference_in_differences(bad)

    with pytest.raises(ValueError, match="time_col must be numeric and finite"):
        bad_time = data.copy()
        bad_time["time"] = "not-a-number"
        difference_in_differences(bad_time)

    with pytest.raises(ValueError, match="outcome_col must be numeric and finite"):
        bad_outcome = data.copy()
        bad_outcome["outcome"] = "bad"
        difference_in_differences(bad_outcome)

    with pytest.raises(
        ValueError,
        match="post_col must have observations in both pre and post periods",
    ):
        bad_post = data.copy()
        bad_post["post"] = 1
        difference_in_differences(bad_post)

    with pytest.raises(ValueError, match="group_col must be encoded as 0/1"):
        bad_group = data.copy()
        bad_group["treated_group"] = 2
        difference_in_differences(bad_group)
