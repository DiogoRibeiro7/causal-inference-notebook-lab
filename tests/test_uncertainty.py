from __future__ import annotations

import numpy as np
import pytest

from causal_inference_lab.data_generators import make_confounded_binary_treatment
from causal_inference_lab.estimators import aipw_ate
from causal_inference_lab.uncertainty import bootstrap_ate


def test_bootstrap_ate_is_deterministic_and_returns_bounds() -> None:
    dataset = make_confounded_binary_treatment(n=2_000, seed=55)
    data = dataset.data
    covariates = ["x1", "x2", "x3"]

    first = bootstrap_ate(
        data=data,
        estimator=lambda frame: aipw_ate(frame, covariates),
        n_bootstrap_samples=80,
        seed=99,
        confidence_level=0.95,
    )
    second = bootstrap_ate(
        data=data,
        estimator=lambda frame: aipw_ate(frame, covariates),
        n_bootstrap_samples=80,
        seed=99,
        confidence_level=0.95,
    )

    assert first == second
    assert 0.0 <= first.lower <= first.estimate <= first.upper <= first.estimate + abs(first.estimate) + 1.0
    assert first.n_observations == len(data)
    assert first.n_bootstrap_samples == 80


def test_bootstrap_ate_catches_invalid_inputs() -> None:
    dataset = make_confounded_binary_treatment(n=100, seed=55)

    with pytest.raises(ValueError, match="n_bootstrap_samples must be a positive integer"):
        bootstrap_ate(dataset.data, lambda frame: aipw_ate(frame, ["x1", "x2", "x3"]), n_bootstrap_samples=0)

    with pytest.raises(ValueError, match="n_bootstrap_samples must be at least 2"):
        bootstrap_ate(
            dataset.data,
            lambda frame: aipw_ate(frame, ["x1", "x2", "x3"]),
            n_bootstrap_samples=1,
        )

    with pytest.raises(ValueError, match="confidence_level must be strictly between 0 and 1"):
        bootstrap_ate(
            dataset.data,
            lambda frame: aipw_ate(frame, ["x1", "x2", "x3"]),
            n_bootstrap_samples=10,
            confidence_level=1.1,
        )

    with pytest.raises(TypeError, match="covariates must be a sequence of column names, not a string"):
        bootstrap_ate(
            dataset.data,
            lambda frame, _covariates: aipw_ate(frame, ["x1", "x2", "x3"]),
            covariates="x1",
            n_bootstrap_samples=10,
        )

    with pytest.raises(ValueError, match="covariates must not be empty"):
        bootstrap_ate(
            dataset.data,
            lambda frame, _covariates: aipw_ate(frame, ["x1", "x2", "x3"]),
            covariates=[],
            n_bootstrap_samples=10,
        )

    class BadEstimatorResult:
        def __init__(self) -> None:
            self.estimate = "bad-value"

    def bad_estimator(_: pd.DataFrame) -> BadEstimatorResult:
        return BadEstimatorResult()

    with pytest.raises(TypeError, match="estimator `estimate` must be numeric"):
        bootstrap_ate(
            dataset.data,
            bad_estimator,
            n_bootstrap_samples=3,
            seed=0,
            confidence_level=0.95,
        )
