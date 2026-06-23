from __future__ import annotations

import numpy as np

from causal_inference_lab.data_generators import make_sharp_rdd_data
from causal_inference_lab.rdd import local_linear_rdd, rdd_bandwidth_sensitivity


def test_local_linear_rdd_estimate_close_to_true_ate() -> None:
    dataset = make_sharp_rdd_data(n=4_000, seed=12)
    data = dataset.data

    result = local_linear_rdd(data, cutoff=0.0, bandwidth=1.0)

    assert abs(result.estimate - dataset.true_ate) < 0.8
    assert result.standard_error > 0
    assert result.bandwidth == 1.0


def test_rdd_bandwidth_sensitivity_includes_monotonic_inputs() -> None:
    dataset = make_sharp_rdd_data(n=3_000, seed=13)
    data = dataset.data
    sensitivities = rdd_bandwidth_sensitivity(data, bandwidth_grid=[0.5, 1.0, 1.5])

    assert sensitivities.shape[0] == 3
    assert np.all(np.isfinite(sensitivities["estimate"]))
