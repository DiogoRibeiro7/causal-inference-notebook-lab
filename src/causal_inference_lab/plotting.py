"""Plotting helpers for causal inference notebooks."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_propensity_overlap(
    data: pd.DataFrame,
    propensity_scores: np.ndarray,
    treatment_col: str = "treatment",
) -> plt.Figure:
    """Plot propensity score distributions by treatment group."""

    if len(propensity_scores) != len(data):
        raise ValueError("propensity_scores must have the same length as data.")

    fig, ax = plt.subplots(figsize=(8, 4))
    treated = data[treatment_col].to_numpy() == 1
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

    required = {"covariate", "smd"}
    if not required.issubset(balance.columns):
        raise ValueError(f"balance must contain columns: {required}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ordered = balance.sort_values("smd")
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
