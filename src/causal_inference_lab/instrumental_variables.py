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
) -> None:
    required = [treatment_col, outcome_col, instrument_col]
    if covariates is not None:
        required.extend(covariates)

    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if data.empty:
        raise ValueError("data must not be empty.")

    if data[required].isnull().any().any():
        raise ValueError("Required columns must not contain missing values.")

    if data[instrument_col].nunique() < 2:
        raise ValueError("Instrument must vary across observations.")

    treatment_values = set(data[treatment_col].dropna().unique().tolist())
    if not treatment_values.issubset({0, 1}):
        raise ValueError("treatment must be binary and encoded as 0/1.")

    if data[treatment_col].nunique() < 2:
        raise ValueError("Both treated and control outcomes are required for IV estimation.")


def _build_design_matrix(
    data: pd.DataFrame,
    treatment_col: str,
    covariates: Sequence[str] | None,
) -> pd.DataFrame:
    if covariates is None or len(covariates) == 0:
        return pd.DataFrame(index=data.index)

    return data.loc[:, list(covariates)]


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

    _validate_iv_inputs(
        data=data,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        instrument_col=instrument_col,
        covariates=covariates,
    )

    controls = _build_design_matrix(data, treatment_col=treatment_col, covariates=covariates)
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

    instrument_coef = float(first_stage.params[instrument_col])
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
