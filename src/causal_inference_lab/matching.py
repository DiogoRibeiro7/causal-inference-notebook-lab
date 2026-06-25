"""Matching-based causal estimators and diagnostics."""

from __future__ import annotations

import numbers
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from causal_inference_lab.estimators import EffectEstimate


@dataclass(frozen=True)
class MatchingResult:
    """Matched dataset and ATT estimate metadata."""

    effect: EffectEstimate
    matched_data: pd.DataFrame
    dropped_units: int


def _validate_matching_inputs(
    data: pd.DataFrame,
    covariates: Sequence[str],
    treatment_col: str,
    outcome_col: str | None = None,
) -> list[str]:
    """Validate matching inputs and return normalized covariate names."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("data must not be empty.")

    if not isinstance(treatment_col, str):
        raise TypeError("treatment_col must be a string.")

    if isinstance(covariates, str):
        raise TypeError("covariates must be a sequence of column names, not a string.")
    try:
        covariate_list = list(covariates)
    except TypeError as exc:
        raise TypeError("covariates must be a sequence of column names.") from exc

    if not covariate_list:
        raise ValueError("covariates must not be empty.")
    if not all(isinstance(column, str) for column in covariate_list):
        raise TypeError("covariates must be a sequence of strings.")
    if len(set(covariate_list)) != len(covariate_list):
        raise ValueError("covariates must be unique.")

    required_cols = [treatment_col, *covariate_list]
    if outcome_col is not None:
        if not isinstance(outcome_col, str):
            raise TypeError("outcome_col must be a string.")
        required_cols.append(outcome_col)

    missing = [column for column in required_cols if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    try:
        treatment = data[treatment_col].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("treatment must be numeric and finite.") from exc
    if not np.all(np.isfinite(treatment)):
        raise ValueError("treatment must not contain NaN or infinite values.")
    if np.any(np.isin(treatment, [0, 1], invert=True)):
        raise ValueError("treatment must be binary and encoded as 0/1.")

    if not np.any(treatment == 1) or not np.any(treatment == 0):
        raise ValueError("Both treated and control groups are required for matching.")

    try:
        covariate_matrix = data.loc[:, covariate_list].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("covariates must be numeric and finite.") from exc
    if not np.all(np.isfinite(covariate_matrix)):
        raise ValueError("covariates must not contain NaN or infinite values.")

    if outcome_col is not None:
        try:
            outcome = data[outcome_col].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("outcome must be numeric and finite.") from exc
        if not np.all(np.isfinite(outcome)):
            raise ValueError("outcome must not contain NaN or infinite values.")

    return covariate_list


def _validate_caliper(caliper: float | None, param_name: str = "caliper") -> None:
    """Validate caliper argument used by matchers."""

    if caliper is None:
        return
    if isinstance(caliper, bool) or not isinstance(caliper, numbers.Real):
        raise TypeError(f"{param_name} must be a numeric value.")
    if not np.isfinite(float(caliper)):
        raise ValueError(f"{param_name} must be finite.")
    if caliper < 0:
        raise ValueError(f"{param_name} must be non-negative.")


def _validate_random_state(random_state: int | None) -> None:
    """Validate optional random state used in tie-breaking."""

    if random_state is not None:
        if isinstance(random_state, bool) or not isinstance(random_state, numbers.Integral):
            raise ValueError("random_state must be an integer.")


def _validate_propensity_scores(propensity_scores: np.ndarray, n: int) -> np.ndarray:
    """Validate precomputed propensity scores."""

    scores = np.asarray(propensity_scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("propensity_scores must be one-dimensional.")
    if scores.shape[0] != n:
        raise ValueError("propensity_scores length must match data length.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("propensity_scores must be finite.")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("propensity_scores must be in [0, 1].")
    return scores


def _prepare_treatment_mask(data: pd.DataFrame, treatment_col: str) -> np.ndarray:
    """Construct a boolean treated-mask array."""

    return data[treatment_col].to_numpy(dtype=float) == 1


def _select_matches_with_tiebreak(
    distances: np.ndarray,
    candidates: np.ndarray,
    caliper: float | None,
    random_state: int | None,
) -> int:
    """Select one nearest candidate with optional random tie-breaking."""

    if caliper is not None and caliper > 0:
        candidate_mask = distances <= caliper
        candidates = candidates[candidate_mask]
        distances = distances[candidate_mask]
    if len(candidates) == 0:
        return -1

    min_distance = float(np.min(distances))
    best = np.where(np.isclose(distances, min_distance))[0]
    if random_state is None:
        return int(candidates[best[0]])

    rng = np.random.default_rng(random_state)
    return int(candidates[best[rng.integers(0, len(best))]])


def _build_matched_data(
    data: pd.DataFrame,
    treatment_col: str,
    covariate_matrix: np.ndarray,
    treatment_mask: np.ndarray,
    caliper: float | None,
    random_state: int | None,
) -> tuple[pd.DataFrame, int]:
    """Build paired matches from a one-dimensional matching score."""

    if covariate_matrix.ndim != 1:
        raise ValueError("covariate_matrix must be one-dimensional for propensity-score matching.")

    treated_indices = np.where(treatment_mask)[0]
    control_indices = np.where(~treatment_mask)[0]

    if len(control_indices) == 0:
        raise ValueError("No control units available for matching.")

    matched_rows: list[dict[Hashable, object]] = []
    dropped = 0
    for treated_index in treated_indices:
        control_distances = np.abs(
            covariate_matrix[control_indices] - covariate_matrix[treated_index]
        )
        matched_control_position = _select_matches_with_tiebreak(
            control_distances,
            control_indices,
            caliper=caliper,
            random_state=random_state,
        )
        if matched_control_position < 0:
            dropped += 1
            continue

        control_index = int(matched_control_position)
        pair_id = len(matched_rows) // 2
        treated_row = data.iloc[treated_index].to_dict()
        control_row = data.iloc[control_index].to_dict()
        treated_row["match_pair_id"] = pair_id
        control_row["match_pair_id"] = pair_id
        matched_rows.extend((treated_row, control_row))

    if not matched_rows:
        raise ValueError(
            "Matching failed: every treated unit was unmatched with the current settings."
        )

    matched_data = pd.DataFrame(matched_rows).reset_index(drop=True)
    return matched_data, dropped


def _estimate_att_from_matched_pairs(
    matched_data: pd.DataFrame,
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
) -> float:
    """Estimate ATT from one-to-one matched pairs."""

    pair_effects: list[float] = []
    for _, pair in matched_data.groupby("match_pair_id"):
        treated = pair.loc[pair[treatment_col] == 1, outcome_col].to_numpy(dtype=float)
        control = pair.loc[pair[treatment_col] == 0, outcome_col].to_numpy(dtype=float)
        if len(treated) != 1 or len(control) != 1:
            continue
        pair_effects.append(float(treated[0] - control[0]))

    if not pair_effects:
        raise ValueError("No complete matched pairs were formed.")

    return float(np.mean(pair_effects))


def propensity_score_matching(
    data: pd.DataFrame,
    covariates: Sequence[str],
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    propensity_scores: np.ndarray | None = None,
    caliper: float | None = None,
    random_state: int | None = None,
) -> MatchingResult:
    """Build matched pairs using nearest-neighbour matching on propensity scores.

    This matcher uses replacement, so controls can be reused across treated units.

    Args:
        data: Input DataFrame containing treatment and outcome columns.
        covariates: Covariate names used in the treatment model.
        treatment_col: Binary treatment column.
        outcome_col: Outcome column.
        propensity_scores: Optional pre-computed propensity scores.
        caliper: Optional max absolute distance on the matching score.
        random_state: Seed used for deterministic tiebreaks.

    Returns:
        MatchingResult with matched rows, ATT estimate, and dropped units.
    """

    covariate_list = _validate_matching_inputs(
        data, covariates, treatment_col=treatment_col, outcome_col=outcome_col
    )
    _validate_caliper(caliper)
    _validate_random_state(random_state)

    if propensity_scores is None:
        from causal_inference_lab.estimators import estimate_propensity_scores

        propensity_scores = estimate_propensity_scores(
            data, covariate_list, treatment_col=treatment_col
        )

    propensity_scores = _validate_propensity_scores(propensity_scores, len(data))
    treatment_mask = _prepare_treatment_mask(data, treatment_col)

    matched_data, dropped = _build_matched_data(
        data=data,
        treatment_col=treatment_col,
        covariate_matrix=propensity_scores,
        treatment_mask=treatment_mask,
        caliper=caliper,
        random_state=random_state,
    )
    estimate = float(
        _estimate_att_from_matched_pairs(
            matched_data, treatment_col=treatment_col, outcome_col=outcome_col
        )
    )

    return MatchingResult(
        effect=EffectEstimate(
            estimate=estimate,
            estimator="propensity_score_matching",
            estimand="ATT",
            n_observations=int(len(matched_data)),
        ),
        matched_data=matched_data,
        dropped_units=dropped,
    )


def nearest_neighbour_matching(
    data: pd.DataFrame,
    covariates: Sequence[str],
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    n_neighbors: int = 1,
    caliper: float | None = None,
) -> MatchingResult:
    """Match treated units to nearest control units in standardized covariate space.

    Args:
        data: Input DataFrame containing treatment and outcome columns.
        covariates: Covariate names for distance computation.
        treatment_col: Binary treatment column.
        outcome_col: Outcome column.
        n_neighbors: Number of candidate neighbors searched (1 is nearest neighbor).
        caliper: Optional maximum Euclidean distance for match acceptance.

    Returns:
        MatchingResult with matched rows, ATT estimate, and dropped units.
    """

    covariate_list = _validate_matching_inputs(
        data, covariates, treatment_col=treatment_col, outcome_col=outcome_col
    )
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, numbers.Integral):
        raise ValueError("n_neighbors must be an integer >= 1.")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be an integer >= 1.")
    n_neighbors = int(n_neighbors)
    _validate_caliper(caliper)

    treatment_mask = _prepare_treatment_mask(data, treatment_col)

    scaler = StandardScaler()
    covariate_matrix = scaler.fit_transform(data.loc[:, covariate_list].to_numpy(dtype=float))

    treated_indices = np.where(treatment_mask)[0]
    control_indices = np.where(~treatment_mask)[0]

    if len(control_indices) == 0:
        raise ValueError("No control units available for matching.")

    treated_matrix = covariate_matrix[treated_indices]
    control_matrix = covariate_matrix[control_indices]
    matcher = NearestNeighbors(n_neighbors=min(n_neighbors, len(control_indices)))
    matcher.fit(control_matrix)
    distances, nearest = matcher.kneighbors(treated_matrix, return_distance=True)

    matched_rows: list[dict[Hashable, object]] = []
    dropped = 0
    for treated_index, neighbor_indices, neighbor_distances in zip(
        treated_indices, nearest, distances, strict=False
    ):
        if caliper is not None and neighbor_distances[0] > caliper:
            dropped += 1
            continue
        selected_control = int(control_indices[neighbor_indices[0]])
        pair_id = len(matched_rows) // 2
        treated_row = data.iloc[treated_index].to_dict()
        control_row = data.iloc[selected_control].to_dict()
        treated_row["match_pair_id"] = pair_id
        control_row["match_pair_id"] = pair_id
        matched_rows.extend((treated_row, control_row))

    if not matched_rows:
        raise ValueError(
            "Matching failed: every treated unit was unmatched with the current settings."
        )

    matched_data = pd.DataFrame(matched_rows).reset_index(drop=True)
    estimate = float(
        _estimate_att_from_matched_pairs(
            matched_data, treatment_col=treatment_col, outcome_col=outcome_col
        )
    )

    return MatchingResult(
        effect=EffectEstimate(
            estimate=estimate,
            estimator="nearest_neighbour_matching",
            estimand="ATT",
            n_observations=int(len(matched_data)),
        ),
        matched_data=matched_data,
        dropped_units=dropped,
    )


def matching_balance_table(
    data: pd.DataFrame,
    covariates: Sequence[str],
    matched_data: pd.DataFrame,
    treatment_col: str = "treatment",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare standardized mean differences before and after matching."""

    from causal_inference_lab.diagnostics import balance_table

    _validate_matching_inputs(data, covariates, treatment_col=treatment_col)
    if not isinstance(matched_data, pd.DataFrame):
        raise TypeError("matched_data must be a pandas DataFrame.")
    if matched_data.empty:
        raise ValueError("matched_data must not be empty.")
    _validate_matching_inputs(matched_data, covariates, treatment_col=treatment_col)

    before = balance_table(data=data, covariates=covariates, treatment_col=treatment_col)
    after = balance_table(data=matched_data, covariates=covariates, treatment_col=treatment_col)
    return before, after
