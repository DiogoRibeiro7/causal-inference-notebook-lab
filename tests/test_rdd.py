from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_inference_lab.data_generators import make_sharp_rdd_data
from causal_inference_lab.rdd import local_linear_rdd, rdd_bandwidth_sensitivity


def test_local_linear_rdd_estimate_close_to_true_ate() -> None:
    dataset = make_sharp_rdd_data(n=4_000, seed=12)
    data = dataset.data

    result = local_linear_rdd(data, cutoff=0.0, bandwidth=1.0)

    assert abs(result.estimate - dataset.true_ate) < 0.8
    assert result.standard_error > 0
    assert result.bandwidth == 1.0


def test_local_linear_rdd_accepts_numpy_numeric_cutoff_and_bandwidth() -> None:
    dataset = make_sharp_rdd_data(n=2_000, seed=12)
    data = dataset.data

    float_result = local_linear_rdd(data, cutoff=np.float64(0.0), bandwidth=np.float64(1.0))
    int_result = local_linear_rdd(data, cutoff=np.int64(0), bandwidth=np.int64(1))

    assert np.isfinite(float_result.estimate)
    assert np.isfinite(float_result.standard_error)
    assert np.isfinite(int_result.estimate)
    assert np.isfinite(int_result.standard_error)


def test_rdd_bandwidth_sensitivity_includes_monotonic_inputs() -> None:
    dataset = make_sharp_rdd_data(n=3_000, seed=13)
    data = dataset.data
    sensitivities = rdd_bandwidth_sensitivity(
        data,
        bandwidth_grid=[np.float64(0.5), np.float64(1.0), np.float64(1.5)],
    )

    assert sensitivities.shape[0] == 3
    assert np.all(np.isfinite(sensitivities["estimate"]))

    int_sensitivities = rdd_bandwidth_sensitivity(
        data,
        bandwidth_grid=[np.int64(1), np.int64(2)],
    )
    assert int_sensitivities.shape[0] == 2
    assert np.all(np.isfinite(int_sensitivities["estimate"]))


def test_local_linear_rdd_input_validation() -> None:
    with pytest.raises(ValueError, match="data must be a pandas DataFrame."):
        local_linear_rdd("bad")

    with pytest.raises(ValueError, match="running_col must be a string."):
        local_linear_rdd(
            make_sharp_rdd_data(n=100, seed=1).data,
            running_col=123,
        )

    with pytest.raises(ValueError, match="Missing required columns"):
        local_linear_rdd(pd.DataFrame({"running": [0, 1], "treatment": [0, 1]}), outcome_col="outcome")

    with pytest.raises(ValueError, match="must be numeric"):
        local_linear_rdd(
            pd.DataFrame(
                {
                    "running": [0, 1],
                    "outcome": [1.0, 2.0],
                    "treatment": ["yes", "no"],
                }
            )
        )

    with pytest.raises(ValueError, match="binary"):
        local_linear_rdd(
            pd.DataFrame(
                {"running": [0.0, 1.0], "outcome": [1.0, 2.0], "treatment": [0.0, 2.0]}
            )
        )

    with pytest.raises(ValueError, match="must be a finite real number."):
        local_linear_rdd(make_sharp_rdd_data(n=50, seed=2).data, cutoff=np.nan)

    with pytest.raises(ValueError, match="bandwidth must be positive."):
        local_linear_rdd(make_sharp_rdd_data(n=50, seed=3).data, bandwidth=-1.0)

    with pytest.raises(ValueError, match="cutoff must be a finite real number."):
        local_linear_rdd(make_sharp_rdd_data(n=50, seed=4).data, cutoff=True)

    with pytest.raises(TypeError, match="bandwidth_grid must be a sequence of numeric bandwidths."):
        rdd_bandwidth_sensitivity(make_sharp_rdd_data(n=100, seed=4).data, bandwidth_grid="bad-grid")

    with pytest.raises(TypeError, match="bandwidth_grid must be a sequence of numeric bandwidths."):
        rdd_bandwidth_sensitivity(make_sharp_rdd_data(n=100, seed=5).data, bandwidth_grid=[1.0, True])

    with pytest.raises(ValueError, match="all bandwidths must be finite positive values."):
        rdd_bandwidth_sensitivity(
            make_sharp_rdd_data(n=100, seed=6).data,
            bandwidth_grid=[1.0, float("inf")],
        )
