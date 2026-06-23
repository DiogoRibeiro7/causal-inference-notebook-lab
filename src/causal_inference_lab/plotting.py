"""Plotting helpers for causal inference notebooks."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _validate_data_frame(data: pd.DataFrame) -> None:
    """Validate DataFrame inputs for plotting utilities."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("data must not be empty.")


def _validate_treatment_series(data: pd.DataFrame, treatment_col: str) -> np.ndarray:
    """Validate a binary treatment column and return a float-treated mask."""

    if not isinstance(treatment_col, str):
        raise TypeError("treatment_col must be a string.")
    if treatment_col not in data.columns:
        raise ValueError(f"Missing required columns: ['{treatment_col}']")

    try:
        treatment = data[treatment_col].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("treatment must be numeric and finite.") from exc

    if not np.all(np.isfinite(treatment)):
        raise ValueError("treatment must be numeric and finite.")
    if np.any(np.isin(treatment, [0, 1], invert=True)):
        raise ValueError("treatment must be binary and encoded as 0/1.")
    if not np.any(treatment == 1) or not np.any(treatment == 0):
        raise ValueError("Both treated and control groups are required.")

    return treatment == 1


def _validate_binary_group_column(data: pd.DataFrame, group_col: str) -> np.ndarray:
    """Validate a binary grouping column and return a float group mask."""

    if not isinstance(group_col, str):
        raise TypeError("group_col must be a string.")
    if group_col not in data.columns:
        raise ValueError(f"Missing required columns: ['{group_col}']")

    try:
        groups = data[group_col].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("group_col must be numeric and finite.") from exc

    if not np.all(np.isfinite(groups)):
        raise ValueError("group_col must be numeric and finite.")
    if np.any(np.isin(groups, [0, 1], invert=True)):
        raise ValueError("group_col must be binary and encoded as 0/1.")
    if not np.any(groups == 1) or not np.any(groups == 0):
        raise ValueError("Both treated and control groups are required.")

    return groups == 1


def _validate_numeric_array(values: np.ndarray, name: str) -> np.ndarray:
    """Validate a numeric one-dimensional finite array."""

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must not contain NaN or infinite values.")
    if len(arr) == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr


def plot_propensity_overlap(
    data: pd.DataFrame,
    propensity_scores: np.ndarray,
    treatment_col: str = "treatment",
) -> plt.Figure:
    """Plot propensity score distributions by treatment group."""

    _validate_data_frame(data)
    treated = _validate_treatment_series(data, treatment_col)

    propensity_scores = _validate_numeric_array(propensity_scores, "propensity_scores")
    if len(propensity_scores) != len(data):
        raise ValueError("propensity_scores must have the same length as data.")
    if np.any((propensity_scores < 0.0) | (propensity_scores > 1.0)):
        raise ValueError("propensity_scores must be in [0, 1].")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(propensity_scores[~treated], bins=30, alpha=0.6, label="Control")
    ax.hist(propensity_scores[treated], bins=30, alpha=0.6, label="Treated")
    ax.set_title("Propensity score overlap")
    ax.set_xlabel("Estimated propensity score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_balance_table(balance: pd.DataFrame) -> plt.Figure:
    """Plot standardized mean differences."""

    _validate_data_frame(balance)

    required = {"covariate", "smd"}
    if not required.issubset(balance.columns):
        raise ValueError(f"balance must contain columns: {required}")

    smd = _validate_numeric_array(balance["smd"].to_numpy(), "smd values")

    fig, ax = plt.subplots(figsize=(8, 4))
    ordered = balance.assign(smd=smd).sort_values("smd")
    ax.barh(ordered["covariate"], ordered["smd"])
    ax.axvline(0.0, linewidth=1)
    ax.axvline(0.1, linestyle="--", linewidth=1)
    ax.axvline(-0.1, linestyle="--", linewidth=1)
    ax.set_title("Covariate balance")
    ax.set_xlabel("Standardized mean difference")
    fig.tight_layout()
    return fig


def plot_did_trends(
    data: pd.DataFrame,
    time_col: str = "time",
    group_col: str = "treated_group",
    outcome_col: str = "outcome",
) -> plt.Figure:
    """Plot average panel trends by treatment group."""

    _validate_data_frame(data)

    if not isinstance(time_col, str):
        raise TypeError("time_col must be a string.")
    if not isinstance(group_col, str):
        raise TypeError("group_col must be a string.")
    if not isinstance(outcome_col, str):
        raise TypeError("outcome_col must be a string.")

    required = [time_col, group_col, outcome_col]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    _ = _validate_binary_group_column(data, group_col)
    _validate_numeric_array(data[outcome_col].to_numpy(), f"{outcome_col}")

    summary = (
        data.groupby([time_col, group_col], as_index=False)[outcome_col]
        .mean()
        .sort_values([group_col, time_col])
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    for group_value, frame in summary.groupby(group_col):
        label = "Treated group" if group_value == 1 else "Control group"
        ax.plot(frame[time_col], frame[outcome_col], marker="o", label=label)

    ax.set_title("Difference-in-differences trends")
    ax.set_xlabel("Time")
    ax.set_ylabel("Average outcome")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cate_recovery(true_cate: Sequence[float], estimated_cate: Sequence[float]) -> plt.Figure:
    """Plot estimated CATE against true CATE."""

    true_array = np.asarray(true_cate, dtype=float)
    estimated_array = np.asarray(estimated_cate, dtype=float)
    if true_array.shape != estimated_array.shape:
        raise ValueError("true_cate and estimated_cate must have the same shape.")
    if true_array.ndim != 1 or estimated_array.ndim != 1:
        raise ValueError("true_cate and estimated_cate must be one-dimensional arrays.")
    if len(true_array) == 0:
        raise ValueError("true_cate and estimated_cate must not be empty.")
    if not np.all(np.isfinite(true_array)):
        raise ValueError("true_cate must be finite.")
    if not np.all(np.isfinite(estimated_array)):
        raise ValueError("estimated_cate must be finite.")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_array, estimated_array, alpha=0.4)
    lower = min(float(true_array.min()), float(estimated_array.min()))
    upper = max(float(true_array.max()), float(estimated_array.max()))
    ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set_title("CATE recovery")
    ax.set_xlabel("True CATE")
    ax.set_ylabel("Estimated CATE")
    fig.tight_layout()
    return fig
