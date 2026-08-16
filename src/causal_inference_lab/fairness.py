"""Allocation fairness for targeting policies.

A targeting rule that ranks people by estimated benefit is often described as
neutral: it uses no protected attribute, only predicted effect. That describes
the *inputs*. It says nothing about the allocation that results, which can be
markedly uneven — and, as notebook 13 shows, can be uneven even when the groups
benefit identically, because estimated benefit reflects measurement quality as
well as need.

These functions measure the allocation a policy produces and construct a
parity-constrained alternative, so the trade-off between total benefit and even
allocation can be quantified rather than asserted.

Nothing here decides what is fair. Demographic parity is one criterion among
several, it can conflict with others, and choosing it is a normative decision
that belongs to the people affected, not to a library.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _validate_inputs(scores: np.ndarray, groups: np.ndarray) -> None:
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional.")
    if groups.ndim != 1:
        raise ValueError("groups must be one-dimensional.")
    if len(scores) != len(groups):
        raise ValueError("scores and groups must have the same length.")
    if len(scores) == 0:
        raise ValueError("scores must not be empty.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite.")


def _validate_budget(budget: int, population: int) -> None:
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
        raise TypeError("budget must be an integer.")
    if budget < 0:
        raise ValueError("budget must not be negative.")
    if budget > population:
        raise ValueError("budget must not exceed the population size.")


def allocation_table(
    groups: Sequence[object],
    selected: Sequence[int],
    true_effects: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Summarise which groups a policy actually treats.

    Args:
        groups: Group label for every unit in the population.
        selected: Indices of the units the policy treats.
        true_effects: Optional per-unit effects, used to report the benefit
            realised within each group.

    Returns:
        One row per group with its population share, how many of its members
        were selected, the share of that group treated (the allocation rate),
        and its share of the treated pool.

    Raises:
        ValueError: If ``selected`` contains an index outside the population.
    """
    group_array = np.asarray(groups)
    population = len(group_array)

    selected_index = np.asarray(selected, dtype=int)
    if selected_index.size and (selected_index.min() < 0 or selected_index.max() >= population):
        raise ValueError("selected contains an index outside the population.")

    chosen = np.zeros(population, dtype=bool)
    chosen[selected_index] = True

    rows = []
    for label in pd.unique(group_array):
        in_group = group_array == label
        treated_in_group = int(np.sum(in_group & chosen))
        row = {
            "group": label,
            "population": int(np.sum(in_group)),
            "population_share": float(np.mean(in_group)),
            "treated": treated_in_group,
            # Allocation rate: of this group, what fraction got the treatment.
            "allocation_rate": float(treated_in_group / np.sum(in_group)),
            # Share of treated: of everyone treated, what fraction came from here.
            "share_of_treated": float(treated_in_group / max(chosen.sum(), 1)),
        }
        if true_effects is not None:
            effects = np.asarray(true_effects, dtype=float)
            row["mean_true_effect"] = float(np.mean(effects[in_group]))
            row["realised_benefit"] = float(np.sum(effects[in_group & chosen]))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def allocation_disparity(groups: Sequence[object], selected: Sequence[int]) -> float:
    """Largest gap in allocation rate between any two groups.

    Zero means every group is treated at the same rate — demographic parity.
    The measure says nothing about whether that is the right target.

    Args:
        groups: Group label for every unit in the population.
        selected: Indices of the units the policy treats.

    Returns:
        The difference between the highest and lowest group allocation rates.
    """
    rates = allocation_table(groups, selected)["allocation_rate"]
    return float(rates.max() - rates.min())


def parity_constrained_selection(
    scores: Sequence[float],
    groups: Sequence[object],
    budget: int,
) -> np.ndarray:
    """Select a budget so that every group is treated at the same rate.

    Each group receives a quota proportional to its size, and the highest
    scoring members within each group fill it. The ranking is therefore still
    used — but only *within* groups, so it can no longer move treatment between
    them.

    Rounding quotas to whole people can leave the budget slightly unfilled; any
    remainder goes to the highest scorers not yet chosen, which reintroduces a
    small amount of disparity. That is a real property of integer allocation,
    not an approximation error to be hidden.

    Args:
        scores: Estimated benefit for every unit.
        groups: Group label for every unit.
        budget: Number of units to treat.

    Returns:
        Indices of the selected units.
    """
    score_array = np.asarray(scores, dtype=float)
    group_array = np.asarray(groups)
    _validate_inputs(score_array, group_array)
    _validate_budget(budget, len(score_array))

    if budget == 0:
        return np.array([], dtype=int)

    chosen: list[int] = []
    for label in pd.unique(group_array):
        member_index = np.flatnonzero(group_array == label)
        quota = int(round(budget * len(member_index) / len(score_array)))
        quota = min(quota, len(member_index))
        if quota <= 0:
            continue
        ranked = member_index[np.argsort(-score_array[member_index])]
        chosen.extend(ranked[:quota].tolist())

    # Fill or trim to hit the budget exactly.
    if len(chosen) > budget:
        chosen = sorted(chosen, key=lambda i: -score_array[i])[:budget]
    elif len(chosen) < budget:
        remaining = [i for i in np.argsort(-score_array) if i not in set(chosen)]
        chosen.extend(remaining[: budget - len(chosen)])

    return np.array(sorted(chosen), dtype=int)


def top_k_selection(scores: Sequence[float], budget: int) -> np.ndarray:
    """Select the highest-scoring units, ignoring group membership.

    Args:
        scores: Estimated benefit for every unit.
        budget: Number of units to treat.

    Returns:
        Indices of the selected units.
    """
    score_array = np.asarray(scores, dtype=float)
    _validate_inputs(score_array, np.zeros(len(score_array)))
    _validate_budget(budget, len(score_array))
    if budget == 0:
        return np.array([], dtype=int)
    return np.sort(np.argsort(-score_array)[:budget])
