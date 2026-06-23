from __future__ import annotations

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
