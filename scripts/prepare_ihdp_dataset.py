"""Prepare the IHDP semi-synthetic benchmark for notebook 12.

IHDP takes covariates and treatment assignment from the real Infant Health and
Development Program, then *simulates* the outcomes. That combination is what
makes it useful: the confounding structure is real, but the true individual
effects are known, so an estimator can be scored rather than merely compared
against other estimators.

It complements the Lalonde benchmark in notebook 10, where the truth is known
only as an aggregate experimental figure and every observational estimator
fails to recover it.

Ten replications are downloaded. Each redraws the simulated outcomes from the
same covariates, so the spread of an estimator's error across replications says
far more about it than its error on any single draw — which is the point the
notebook builds on, because on replication 1 the unadjusted estimate happens to
be almost exactly right.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

DEFAULT_OUTPUT = Path("data") / "processed" / "ihdp_benchmark.csv"
DEFAULT_MANIFEST = Path("data") / "processed" / "ihdp_benchmark_manifest.json"

SOURCE_TEMPLATE = (
    "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/"
    "datasets/IHDP/csv/ihdp_npci_{replication}.csv"
)
N_REPLICATIONS = 10
N_COVARIATES = 25

# The published CSVs carry no header. This is the documented column order.
RAW_COLUMNS = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [
    f"x{index}" for index in range(1, N_COVARIATES + 1)
]
COVARIATES = [f"x{index}" for index in range(1, N_COVARIATES + 1)]


@dataclass(frozen=True)
class IHDPPreparationResult:
    """Result metadata from the IHDP preparation script."""

    path: Path
    rows: int
    replications: int
    source: str
    columns: list[str]


def _read_replication(replication: int) -> pd.DataFrame:
    """Download one replication and label its columns."""
    url = SOURCE_TEMPLATE.format(replication=replication)
    request = Request(url, headers={"User-Agent": "causal-inference-notebook-lab"})
    try:
        with urlopen(request, timeout=60) as response:
            payload = response.read().decode("utf-8")
    except (HTTPError, URLError, ValueError) as err:
        raise RuntimeError(
            f"Unable to download IHDP replication {replication} from {url}. "
            "Check network access and retry."
        ) from err

    frame = pd.read_csv(io.StringIO(payload), header=None)
    if frame.shape[1] != len(RAW_COLUMNS):
        raise ValueError(
            f"Replication {replication} has {frame.shape[1]} columns, "
            f"expected {len(RAW_COLUMNS)}. The upstream format may have changed."
        )
    frame.columns = RAW_COLUMNS
    return frame


def _normalize(frame: pd.DataFrame, replication: int) -> pd.DataFrame:
    """Add the derived columns the notebooks expect."""
    data = frame.copy()

    data["treatment"] = data["treatment"].astype(int)
    # `outcome` is the factual outcome, the only one an analyst would observe.
    data["outcome"] = data["y_factual"].astype(float)
    # mu1 and mu0 are the simulated conditional means, so their difference is
    # the true individual effect. Nothing but validation may use these.
    data["true_ite"] = (data["mu1"] - data["mu0"]).astype(float)
    data["replication"] = replication

    ordered = ["replication", "treatment", "outcome", "true_ite", *COVARIATES]
    return data[ordered + ["y_factual", "y_cfactual", "mu0", "mu1"]]


def prepare_ihdp_dataset(
    output_path: Path = DEFAULT_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST,
    replications: int = N_REPLICATIONS,
) -> IHDPPreparationResult:
    """Download, clean, and write the IHDP benchmark.

    Args:
        output_path: Where to save the combined CSV.
        manifest_path: Where to save a small metadata manifest.
        replications: How many replications to fetch, starting at 1.

    Returns:
        Metadata for reproducibility.

    Raises:
        ValueError: If ``replications`` is not a positive integer.
    """
    if not isinstance(replications, int) or isinstance(replications, bool):
        raise TypeError("replications must be an integer.")
    if replications < 1:
        raise ValueError("replications must be at least 1.")

    output_path = Path(output_path)
    manifest_path = Path(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = [_normalize(_read_replication(index), index) for index in range(1, replications + 1)]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_path, index=False)

    result = IHDPPreparationResult(
        path=output_path,
        rows=int(combined.shape[0]),
        replications=replications,
        source=SOURCE_TEMPLATE.format(replication=f"{{1..{replications}}}"),
        columns=combined.columns.tolist(),
    )

    manifest: dict[str, Any] = {
        "source": result.source,
        "description": (
            "IHDP semi-synthetic benchmark. Covariates and treatment assignment come "
            "from the Infant Health and Development Program; outcomes are simulated, "
            "so true individual effects are known."
        ),
        "rows": result.rows,
        "replications": replications,
        "rows_per_replication": int(combined.shape[0] / replications),
        "columns": result.columns,
        "outcome_column": "outcome",
        "treatment_column": "treatment",
        "covariates": COVARIATES,
        "ground_truth_columns": ["true_ite", "mu0", "mu1", "y_cfactual"],
        "ground_truth_note": (
            "Ground-truth columns are for validation only. An estimator that reads "
            "them is answering a question no real analysis can ask."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return result


if __name__ == "__main__":
    outcome = prepare_ihdp_dataset()
    print(f"Prepared IHDP benchmark at: {outcome.path}")
    print(f"Rows: {outcome.rows:,} across {outcome.replications} replications")
