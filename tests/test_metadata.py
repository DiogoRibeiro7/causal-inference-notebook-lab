"""Guard the release metadata that no other test touches.

The version lives in four places that a release must keep in agreement: the
package, the installed distribution, the citation file, and the Zenodo deposit
metadata. Nothing but this test notices when one of them drifts, and the cost of
drift is a published DOI whose recorded version is wrong.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import pytest
import yaml

import causal_inference_lab

REPO_ROOT = Path(__file__).resolve().parents[1]
CITATION_PATH = REPO_ROOT / "CITATION.cff"
ZENODO_PATH = REPO_ROOT / ".zenodo.json"


def _citation() -> dict[str, Any]:
    loaded = yaml.safe_load(CITATION_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _zenodo() -> dict[str, Any]:
    loaded = json.loads(ZENODO_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_citation_file_version_matches_package() -> None:
    assert _citation()["version"] == causal_inference_lab.__version__


def test_zenodo_version_matches_package() -> None:
    assert _zenodo()["version"] == causal_inference_lab.__version__


def test_distribution_version_matches_package() -> None:
    """The dynamic version in pyproject.toml resolves to ``__version__``."""
    try:
        installed = distribution_version("causal-inference-notebook-lab")
    except PackageNotFoundError:  # pragma: no cover - only when not installed
        pytest.skip("package is not installed; run `pip install -e .`")
    assert installed == causal_inference_lab.__version__


def test_citation_and_zenodo_agree_on_authorship() -> None:
    citation_author = _citation()["authors"][0]
    zenodo_creator = _zenodo()["creators"][0]

    expected_name = f"{citation_author['family-names']}, {citation_author['given-names']}"
    assert zenodo_creator["name"] == expected_name

    # CITATION.cff stores the ORCID as a URL, Zenodo as a bare identifier.
    assert citation_author["orcid"].endswith(zenodo_creator["orcid"])


def test_citation_and_zenodo_agree_on_license() -> None:
    assert _citation()["license"] == _zenodo()["license"]


def test_declared_license_file_exists() -> None:
    assert (REPO_ROOT / "LICENSE").is_file()


def test_changelog_records_the_current_version() -> None:
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{causal_inference_lab.__version__}]" in changelog


def test_package_ships_a_typing_marker() -> None:
    """Without py.typed, downstream type checkers ignore our annotations."""
    marker = Path(causal_inference_lab.__file__).parent / "py.typed"
    assert marker.is_file()


def test_public_api_is_importable() -> None:
    """Every name in __all__ resolves, so the documented API is not a fiction."""
    missing = [
        name for name in causal_inference_lab.__all__ if not hasattr(causal_inference_lab, name)
    ]
    assert not missing, f"exported but missing: {missing}"
