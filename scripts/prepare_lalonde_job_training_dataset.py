"""Prepare the Lalonde / NSW benchmark dataset for notebook 10."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

DEFAULT_OUTPUT = Path("data") / "processed" / "lalonde_job_training.csv"
DEFAULT_MANIFEST = Path("data") / "processed" / "lalonde_job_training_manifest.json"
PUBLIC_URLS = [
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MatchIt/lalonde.csv",
]


@dataclass(frozen=True)
class LalondePreparationResult:
    """Result metadata from the Lalonde preparation script."""

    path: Path
    rows: int
    source: str
    columns: list[str]


def _normalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()

    rename = {
        "treat": "treatment",
        "re74": "earnings_74",
        "re75": "earnings_75",
        "re78": "re78",
    }
    data = data.rename(columns=rename)

    if "race" not in data.columns:
        raise ValueError("Expected a 'race' column in the source dataset.")

    data["black"] = (data["race"] == "black").astype(int)
    data["hispanic"] = (data["race"] == "hispan").astype(int)
    data["education"] = data["educ"].astype(float)
    data = data.drop(columns=["educ", "race", "rownames"], errors="ignore")

    for outcome_col in ["re78", "earnings_74", "earnings_75"]:
        data[outcome_col] = pd.to_numeric(data[outcome_col], errors="coerce")

    data["u74"] = (data["earnings_74"] == 0.0).astype(int)
    data["u75"] = (data["earnings_75"] == 0.0).astype(int)

    data["treatment"] = data["treatment"].astype(int)
    data["outcome"] = data["re78"].astype(float)
    data = data.dropna().reset_index(drop=True)

    return data


def _read_public_csv() -> pd.DataFrame:
    last_error: Exception | None = None
    for source in PUBLIC_URLS:
        try:
            with urlopen(source, timeout=30) as response:
                return pd.read_csv(response)
        except (HTTPError, URLError, ValueError) as err:
            last_error = err
    raise RuntimeError(
        "Unable to download the Lalonde dataset from configured public URLs. "
        "Check network access and retry."
    ) from last_error


def prepare_lalonde_job_training_dataset(
    output_path: Path = DEFAULT_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> LalondePreparationResult:
    """Download, clean, and write the benchmark dataset.

    Args:
        output_path: Where to save the prepared CSV.
        manifest_path: Where to save a small metadata manifest.

    Returns:
        Metadata for reproducibility.
    """

    output_path = Path(output_path)
    manifest_path = Path(manifest_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw = _read_public_csv()
    prepared = _normalize_columns(raw)

    # Canonical columns used by notebook 10.
    prepared = prepared[
        [
            "treatment",
            "age",
            "education",
            "black",
            "hispanic",
            "married",
            "nodegree",
            "earnings_74",
            "earnings_75",
            "re78",
            "u74",
            "u75",
            "outcome",
        ]
    ]

    prepared["treatment"] = prepared["treatment"].astype(int)
    prepared.to_csv(output_path, index=False)

    result = LalondePreparationResult(
        path=output_path,
        rows=int(prepared.shape[0]),
        source=PUBLIC_URLS[0],
        columns=prepared.columns.tolist(),
    )

    manifest: dict[str, Any] = {
        "source": result.source,
        "rows": result.rows,
        "columns": result.columns,
        "outcome_column": "outcome",
        "treatment_column": "treatment",
        "covariates": [
            "age",
            "education",
            "black",
            "hispanic",
            "married",
            "nodegree",
            "earnings_74",
            "earnings_75",
            "u74",
            "u75",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return result


if __name__ == "__main__":
    result = prepare_lalonde_job_training_dataset()
    print(f"Prepared Lalonde dataset at: {result.path}")
    print(f"Rows: {result.rows}")
