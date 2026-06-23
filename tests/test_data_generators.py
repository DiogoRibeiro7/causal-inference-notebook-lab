from __future__ import annotations

import numpy as np
import pytest

from causal_inference_lab import data_generators as generators


def test_data_generators_validate_positive_integer_parameters() -> None:
    with pytest.raises(TypeError, match="n must be an integer."):
        generators.make_confounded_binary_treatment(n=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="n must be positive."):
        generators.make_heterogeneous_treatment_data(n=0)

    with pytest.raises(TypeError, match="seed must be an integer."):
        generators.make_iv_data(seed="7")  # type: ignore[arg-type]

    dataset = generators.make_heterogeneous_treatment_data(n=np.int64(10), seed=np.int64(22))
    assert len(dataset.data) == 10


def test_data_generators_validate_did_inputs() -> None:
    with pytest.raises(TypeError, match="n_units must be an integer."):
        generators.make_did_panel(n_units=5.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="n_periods must be at least 4"):
        generators.make_did_panel(n_units=20, n_periods=3)


def test_data_generators_validate_synthetic_control_inputs() -> None:
    with pytest.raises(TypeError, match="treated_unit must be an integer."):
        generators.make_synthetic_control_data(treated_unit=True)  # type: ignore[arg-type]

    synthetic = generators.make_synthetic_control_data(n_units=10, treated_unit=np.int64(3), seed=np.int64(5))
    assert synthetic.data["treated_unit"].sum() == 1

    with pytest.raises(ValueError, match="treated_unit must be between 0 and n_units - 1."):
        generators.make_synthetic_control_data(n_units=10, treated_unit=10)

    with pytest.raises(ValueError, match="treated_unit must be between 0 and n_units - 1."):
        generators.make_synthetic_control_data(n_units=10, treated_unit=-1)


def test_data_generators_validate_rdd_inputs() -> None:
    with pytest.raises(TypeError, match="cutoff must be a finite real number."):
        generators.make_sharp_rdd_data(cutoff=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cutoff must be finite."):
        generators.make_sharp_rdd_data(cutoff=np.inf)

    rdd = generators.make_sharp_rdd_data(n=np.int64(100), cutoff=np.float64(0.0), seed=np.int64(13))
    assert len(rdd.data) == 100
    assert set(rdd.data["treatment"].unique()) == {0, 1}
