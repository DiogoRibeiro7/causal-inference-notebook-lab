from __future__ import annotations

from causal_inference_lab.data_generators import make_synthetic_control_data
from causal_inference_lab.synthetic_control import fit_synthetic_control


def test_synthetic_control_reproduces_known_effect() -> None:
    dataset = make_synthetic_control_data(n_units=20, n_periods=12, pre_periods=6, effect=3.0, seed=22)
    data = dataset.data
    result = fit_synthetic_control(data, treated_unit=0, pre_period_end=5)

    assert abs(result.estimated_effect - dataset.true_ate) < 1.0
    assert abs(result.weights["weight"].sum() - 1.0) < 1e-6
    assert (result.weights["weight"] >= 0).all()
