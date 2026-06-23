from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_inference_lab.plotting import (
    plot_balance_table,
    plot_cate_recovery,
    plot_did_trends,
    plot_propensity_overlap,
)


def test_plot_propensity_overlap_validates_inputs() -> None:
    data = pd.DataFrame(
        {
            "treatment": [1, 0, 1, 0],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns: \\['treatment'\\]"):
        plot_propensity_overlap(data=pd.DataFrame({"wrong": [1, 0, 1, 0]}), propensity_scores=np.array([0.1, 0.2, 0.3, 0.4]))

    with pytest.raises(ValueError, match="treatment must be binary and encoded as 0/1."):
        plot_propensity_overlap(data=data.assign(treatment=[1, 0, 2, 0]), propensity_scores=np.array([0.1, 0.2, 0.3, 0.4]))

    with pytest.raises(ValueError, match="Both treated and control groups are required."):
        plot_propensity_overlap(data=data.assign(treatment=[1, 1, 1, 1]), propensity_scores=np.array([0.1, 0.2, 0.3, 0.4]))

    with pytest.raises(ValueError, match="propensity_scores must have the same length as data."):
        plot_propensity_overlap(data=data, propensity_scores=np.array([0.1, 0.2]))

    with pytest.raises(ValueError, match="propensity_scores must be in \[0, 1\]."):
        plot_propensity_overlap(data=data, propensity_scores=np.array([0.1, 1.2, 0.3, -0.1]))


def test_plot_propensity_overlap_accepts_valid_inputs_and_returns_figure() -> None:
    data = pd.DataFrame({"treatment": [1, 0, 1, 0]})
    fig = plot_propensity_overlap(data, np.array([0.1, 0.2, 0.8, 0.9]))
    assert fig is not None


def test_plot_balance_table_validates_inputs() -> None:
    balance = pd.DataFrame({"covariate": ["x1", "x2"], "smd": [0.1, np.nan]})
    with pytest.raises(ValueError, match="smd values must not contain NaN or infinite values."):
        plot_balance_table(balance)

    with pytest.raises(ValueError, match="balance must contain columns: \{'covariate', 'smd'\}"):
        plot_balance_table(pd.DataFrame({"covariate": ["x1"], "abs_smd": [0.1]}))

    fig = plot_balance_table(pd.DataFrame({"covariate": ["x1", "x2"], "smd": [0.1, -0.2]}))
    assert fig is not None


def test_plot_did_trends_validates_grouping() -> None:
    data = pd.DataFrame(
        {
            "time": [0, 1, 0, 1],
            "treated_group": [1, 0, 1, 0],
            "outcome": [10.0, 12.0, 11.0, 13.0],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns: \['treated_group'\]"):
        plot_did_trends(data=data.drop(columns=["treated_group"]))

    with pytest.raises(ValueError, match="Both treated and control groups are required."):
        plot_did_trends(data=data.assign(treated_group=[1, 1, 1, 1]), group_col="treated_group")

    with pytest.raises(ValueError, match="outcome must not contain NaN or infinite values."):
        plot_did_trends(data=data.assign(outcome=[10.0, np.nan, 11.0, 13.0]), outcome_col="outcome")

    fig = plot_did_trends(data)
    assert fig is not None


def test_plot_cate_recovery_validates_inputs() -> None:
    with pytest.raises(ValueError, match="true_cate and estimated_cate must have the same shape."):
        plot_cate_recovery(np.array([1.0, 2.0]), np.array([1.0]))

    with pytest.raises(ValueError, match="true_cate and estimated_cate must be one-dimensional arrays."):
        plot_cate_recovery(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]))

    with pytest.raises(ValueError, match="true_cate must be finite."):
        plot_cate_recovery(np.array([1.0, np.inf]), np.array([1.0, 2.0]))

    fig = plot_cate_recovery(np.array([1.0, 2.0]), np.array([1.1, 2.1]))
    assert fig is not None
