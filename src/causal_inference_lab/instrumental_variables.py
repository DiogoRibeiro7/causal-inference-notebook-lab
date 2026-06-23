"""Instrumental variables estimators for local average treatment effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from causal_inference_lab.estimators import EffectEstimate


@dataclass(frozen=True)
class IVResult:
    """Container for a two-stage least squares IV estimate."""

    effect: EffectEstimate
    first_stage_f_stat: float
    first_stage_r2: float
    instrument_coefficient: float
    instrument_is_weak: bool


def _validate_iv_inputs(
    data: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    instrument_col: str,
    covariates: Sequence[str] | None,
) -> list[str]:
    """Validate IV inputs and return normalized covariates."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("data must not be empty.")

    if not isinstance(treatment_col, str):
        raise TypeError("treatment_col must be a string.")
    if not isinstance(outcome_col, str):
        raise TypeError("outcome_col must be a string.")
    if not isinstance(instrument_col, str):
        raise TypeError("instrument_col must be a string.")
    if treatment_col == outcome_col:
        raise ValueError("treatment_col and outcome_col must be different.")
    if treatment_col == instrument_col:
        raise ValueError("treatment_col and instrument_col must be different.")
    if outcome_col == instrument_col:
        raise ValueError("outcome_col and instrument_col must be different.")

    if covariates is None:
        covariate_list: list[str] = []
    else:
        if isinstance(covariates, str):
            raise TypeError("covariates must be a sequence of column names, not a string.")
        try:
            covariate_list = list(covariates)
        except TypeError as exc:
            raise TypeError("covariates must be a sequence of column names.") from exc
        if not all(isinstance(column, str) for column in covariate_list):
            raise TypeError("covariates must be a sequence of strings.")
        if len(set(covariate_list)) != len(covariate_list):
            raise ValueError("covariates must be unique.")

    required = [treatment_col, outcome_col, instrument_col, *covariate_list]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if data[required].isnull().any().any():
        raise ValueError("Required columns must not contain missing values.")

    try:
        treatment = data[treatment_col].astype(float).to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("treatment must be numeric and finite.") from exc
    if not np.all(np.isfinite(treatment)):
        raise ValueError("treatment must be numeric and finite.")
    if not np.array_equal(treatment, treatment.astype(int)):
        raise ValueError("treatment must be binary and encoded as 0/1.")
    if not np.isin(np.unique(treatment), [0.0, 1.0]).all():
        raise ValueError("treatment must be binary and encoded as 0/1.")
    if not np.any(treatment == 1) or not np.any(treatment == 0):
        raise ValueError("Both treated and control outcomes are required for IV estimation.")

    try:
        instrument = data[instrument_col].astype(float).to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("instrument must be numeric and finite.") from exc
    if not np.all(np.isfinite(instrument)):
        raise ValueError("instrument must be numeric and finite.")
    if len(np.unique(instrument)) < 2:
        raise ValueError("Instrument must vary across observations.")

    try:
        outcome = data[outcome_col].astype(float).to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("outcome must be numeric and finite.") from exc
    if not np.all(np.isfinite(outcome)):
        raise ValueError("outcome must be numeric and finite.")

    for covariate in covariate_list:
        try:
            data[covariate].astype(float).to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"control covariate '{covariate}' must be numeric and finite.") from exc

    return covariate_list


def _build_design_matrix(
    data: pd.DataFrame,
    covariates: Sequence[str] | None,
) -> pd.DataFrame:
    if covariates is None or len(covariates) == 0:
        return pd.DataFrame(index=data.index)

    return data.loc[:, list(covariates)].astype(float)


def instrumental_variables_ate(
    data: pd.DataFrame,
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    instrument_col: str = "instrument",
    covariates: Sequence[str] | None = None,
) -> IVResult:
    """Estimate ATE using two-stage least squares with a single endogenous treatment.

    The implementation uses a transparent, hands-on two-stage regression:
    1) treatment regressed on instrument and controls,
    2) outcome regressed on fitted treatment and controls.
    """

    covariate_list = _validate_iv_inputs(
        data=data,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        instrument_col=instrument_col,
        covariates=covariates,
    )
    controls = _build_design_matrix(data, covariates=covariate_list)
    instrument = data[[instrument_col]].astype(float)
    first_stage_x = pd.concat([instrument, controls], axis=1)
    first_stage_x = sm.add_constant(first_stage_x, has_constant="add")

    first_stage = sm.OLS(data[treatment_col].astype(float), first_stage_x).fit()
    treatment_hat = first_stage.predict(first_stage_x)

    if np.std(treatment_hat) <= 0:
        raise ValueError(
            "First-stage fitted treatment has zero variance; instrument is not informative."
        )

    second_stage_x = pd.concat([treatment_hat.rename("treatment_hat"), controls], axis=1)
    second_stage = sm.OLS(
        data[outcome_col].astype(float),
        sm.add_constant(second_stage_x, has_constant="add"),
    ).fit()

    if instrument_col not in first_stage.params.index:
        raise ValueError("First-stage model does not include instrument after fitting.")
    instrument_coef = float(first_stage.params[instrument_col])
    if not np.isfinite(instrument_coef):
        raise ValueError("Instrument coefficient is not finite.")

    treatment_index = "treatment_hat"
    estimate = float(second_stage.params[treatment_index])

    first_stage_f_stat = float(first_stage.t_test(f"{instrument_col} = 0").fvalue)
    weak = first_stage_f_stat < 10.0

    effect = EffectEstimate(
        estimate=estimate,
        estimator="instrumental_variables_2sls",
        estimand="ATE",
        n_observations=int(len(data)),
    )

    return IVResult(
        effect=effect,
        first_stage_f_stat=first_stage_f_stat,
        first_stage_r2=float(first_stage.rsquared),
        instrument_coefficient=instrument_coef,
        instrument_is_weak=weak,
    )
