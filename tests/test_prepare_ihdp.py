"""Tests for the IHDP preparation script's parsing and derived columns.

The download itself is not exercised — a test that hits the network would fail
for reasons unrelated to this repository. What is worth pinning is the part that
would corrupt an analysis silently: the column order of the published CSVs is
positional and undocumented in the file itself, so mislabelling it would produce
a dataset that loads cleanly and means something different.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_ihdp_dataset import (  # noqa: E402
    COVARIATES,
    N_COVARIATES,
    RAW_COLUMNS,
    _normalize,
    prepare_ihdp_dataset,
)


def _raw_frame(rows: int = 4) -> pd.DataFrame:
    """A frame shaped like a published IHDP replication."""
    values = {
        "treatment": [1, 0, 1, 0][:rows],
        "y_factual": [6.0, 2.0, 7.0, 3.0][:rows],
        "y_cfactual": [2.0, 6.0, 3.0, 7.0][:rows],
        "mu0": [2.0, 2.0, 3.0, 3.0][:rows],
        "mu1": [6.0, 6.5, 7.0, 7.5][:rows],
    }
    for index in range(1, N_COVARIATES + 1):
        values[f"x{index}"] = [float(index)] * rows
    return pd.DataFrame(values)


def test_raw_columns_match_the_published_width() -> None:
    assert len(RAW_COLUMNS) == 5 + N_COVARIATES
    assert RAW_COLUMNS[:5] == ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"]


def test_true_ite_is_the_difference_of_the_simulated_surfaces() -> None:
    normalized = _normalize(_raw_frame(), replication=1)
    expected = _raw_frame()["mu1"] - _raw_frame()["mu0"]
    pd.testing.assert_series_equal(
        normalized["true_ite"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_outcome_is_the_factual_outcome() -> None:
    """`outcome` must be what an analyst observes, never the counterfactual."""
    normalized = _normalize(_raw_frame(), replication=1)
    pd.testing.assert_series_equal(
        normalized["outcome"].reset_index(drop=True),
        _raw_frame()["y_factual"].reset_index(drop=True),
        check_names=False,
    )


def test_replication_is_recorded() -> None:
    assert (_normalize(_raw_frame(), replication=7)["replication"] == 7).all()


def test_normalized_frame_keeps_every_covariate() -> None:
    normalized = _normalize(_raw_frame(), replication=1)
    assert [c for c in COVARIATES if c not in normalized.columns] == []
    assert normalized["treatment"].dtype.kind == "i"


def test_replications_must_be_a_positive_integer() -> None:
    with pytest.raises(ValueError):
        prepare_ihdp_dataset(replications=0)
    with pytest.raises(TypeError):
        prepare_ihdp_dataset(replications=True)


@pytest.mark.parametrize("column", ["true_ite", "mu0", "mu1", "y_cfactual"])
def test_ground_truth_columns_are_flagged_in_the_manifest(column: str, tmp_path: Path) -> None:
    """The manifest must mark which columns an estimator may not read."""
    manifest_path = Path("data") / "processed" / "ihdp_benchmark_manifest.json"
    if not manifest_path.exists():
        pytest.skip("benchmark not materialized; run `make data`")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert column in manifest["ground_truth_columns"]
