"""Tests for allocation fairness measurement and parity-constrained targeting."""

from __future__ import annotations

import numpy as np
import pytest

from causal_inference_lab.data_generators import make_service_allocation_population
from causal_inference_lab.fairness import (
    allocation_disparity,
    allocation_table,
    parity_constrained_selection,
    top_k_selection,
)


def test_allocation_table_reports_rates_per_group() -> None:
    groups = np.array([0, 0, 0, 0, 1, 1])
    selected = np.array([0, 1, 4])

    table = allocation_table(groups, selected).set_index("group")

    assert table.loc[0, "population"] == 4
    assert table.loc[0, "treated"] == 2
    assert table.loc[0, "allocation_rate"] == pytest.approx(0.5)
    assert table.loc[1, "allocation_rate"] == pytest.approx(0.5)
    assert table["share_of_treated"].sum() == pytest.approx(1.0)


def test_allocation_table_reports_realised_benefit() -> None:
    groups = np.array([0, 0, 1, 1])
    effects = np.array([1.0, 3.0, 2.0, 4.0])

    table = allocation_table(groups, [1, 3], true_effects=effects).set_index("group")

    assert table.loc[0, "realised_benefit"] == pytest.approx(3.0)
    assert table.loc[1, "realised_benefit"] == pytest.approx(4.0)
    assert table.loc[0, "mean_true_effect"] == pytest.approx(2.0)


def test_equal_rates_give_zero_disparity() -> None:
    groups = np.array([0, 0, 1, 1])
    assert allocation_disparity(groups, [0, 2]) == pytest.approx(0.0)


def test_disparity_detects_uneven_allocation() -> None:
    groups = np.array([0, 0, 1, 1])
    # Both treatments go to group 0: rates are 1.0 and 0.0.
    assert allocation_disparity(groups, [0, 1]) == pytest.approx(1.0)


def test_selected_index_must_be_in_range() -> None:
    with pytest.raises(ValueError):
        allocation_table(np.array([0, 1]), [5])


def test_top_k_picks_the_highest_scores() -> None:
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert set(top_k_selection(scores, 2).tolist()) == {1, 3}


def test_parity_selection_equalises_allocation_rates() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat([0, 1], [800, 200])
    # Group 0 scores higher, so an unconstrained rule would favour it.
    scores = rng.normal(np.where(groups == 0, 1.0, 0.0), 0.5)

    budget = 100
    unconstrained = top_k_selection(scores, budget)
    parity = parity_constrained_selection(scores, groups, budget)

    assert len(parity) == budget
    assert allocation_disparity(groups, parity) < allocation_disparity(groups, unconstrained)
    assert allocation_disparity(groups, parity) == pytest.approx(0.0, abs=0.01)


def test_parity_selection_still_ranks_within_groups() -> None:
    """The ranking is not discarded, only prevented from moving treatment between groups."""
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    groups = np.array([0, 0, 1, 1])

    chosen = set(parity_constrained_selection(scores, groups, budget=2).tolist())

    assert chosen == {0, 2}


def test_parity_selection_respects_the_budget_exactly() -> None:
    rng = np.random.default_rng(1)
    groups = rng.integers(0, 3, 500)
    scores = rng.normal(size=500)

    for budget in (0, 1, 7, 123, 500):
        assert len(parity_constrained_selection(scores, groups, budget)) == budget


def test_budget_cannot_exceed_the_population() -> None:
    with pytest.raises(ValueError):
        parity_constrained_selection(np.zeros(5), np.zeros(5), budget=6)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        parity_constrained_selection(np.zeros(5), np.zeros(4), budget=1)


def test_generator_gives_both_groups_the_same_effects_by_default() -> None:
    """With effect_gap zero, any allocation disparity is not about true benefit."""
    data = make_service_allocation_population(n=4_000, seed=3).data

    by_group = data.groupby("group")["true_ite"].mean()

    assert by_group.loc[0] == pytest.approx(by_group.loc[1], abs=0.05)


def test_generator_effect_gap_creates_a_real_difference() -> None:
    data = make_service_allocation_population(n=4_000, effect_gap=0.6, seed=3).data

    by_group = data.groupby("group")["true_ite"].mean()

    assert by_group.loc[0] - by_group.loc[1] == pytest.approx(0.6, abs=0.05)


def test_generator_records_fewer_visits_for_the_under_measured_group() -> None:
    data = make_service_allocation_population(n=4_000, seed=3).data

    visits = data.groupby("group")["prior_visits"].mean()

    assert visits.loc[1] < visits.loc[0]


@pytest.mark.parametrize("share", [0.0, 1.0, -0.1, 1.5])
def test_group_share_must_be_a_proportion(share: float) -> None:
    with pytest.raises(ValueError):
        make_service_allocation_population(n=100, group_share=share)
