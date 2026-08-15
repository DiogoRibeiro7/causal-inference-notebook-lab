"""Enforce the notebook structure that AGENTS.md requires.

A causal analysis is not defensible because it produces a number. It is
defensible because it says what it is estimating, what has to be true for that
estimate to be causal, how the design was checked, how uncertain the answer is,
and what it still cannot conclude.

The repository documented that discipline from the start, but nothing enforced
it, and eight of the twelve notebooks had drifted below it — three were a single
sentence and a ``print``. These tests make the structure a build-time contract:
a notebook that drops its assumptions section fails CI rather than quietly
shipping to the documentation site.

The rules are deliberately about *presence*, not quality. No test can tell you
that an assumptions section is honest. It can tell you that one exists.

Notebooks are committed with their outputs so results are readable on GitHub
without cloning anything. That makes execution state part of the contract: the
tests below check every cell ran, that it ran in order from a fresh kernel, and
that no traceback was committed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"

# The eight steps of the workflow, as section headings, in the order they must
# appear. Estimation cannot precede the estimand; diagnostics cannot precede
# estimation.
REQUIRED_SECTIONS = (
    "Causal question",
    "Data and design",
    "Estimand",
    "Identification assumptions",
    "Estimation",
    "Diagnostics",
    "Uncertainty",
    "Limitations",
)

# Notebook 00 introduces the project and the potential-outcomes vocabulary. It
# estimates nothing, so the analysis contract does not apply to it.
OVERVIEW_NOTEBOOKS = frozenset({"00_project_overview.ipynb"})

MIN_INTERPRETATIONS = 3
INTERPRETATION_MARKER = "**Interpretation.**"


def _notebooks() -> list[Path]:
    found = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    assert found, f"no notebooks found in {NOTEBOOK_DIR}"
    return found


def _load(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _cells(path: Path) -> list[dict[str, Any]]:
    return list(_load(path)["cells"])


def _source(cell: dict[str, Any]) -> str:
    source = cell["source"]
    return source if isinstance(source, str) else "".join(source)


def _markdown_text(path: Path) -> str:
    return "\n".join(_source(c) for c in _cells(path) if c["cell_type"] == "markdown")


def _headings(path: Path) -> list[str]:
    headings = []
    for cell in _cells(path):
        if cell["cell_type"] != "markdown":
            continue
        for line in _source(cell).splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                headings.append(stripped[3:].strip())
    return headings


ALL_NOTEBOOKS = _notebooks()
ANALYSIS_NOTEBOOKS = [p for p in ALL_NOTEBOOKS if p.name not in OVERVIEW_NOTEBOOKS]


def _id(path: Path) -> str:
    return path.name


@pytest.mark.parametrize("notebook", ALL_NOTEBOOKS, ids=_id)
def test_notebook_starts_with_a_title(notebook: Path) -> None:
    cells = _cells(notebook)
    assert cells, f"{notebook.name} is empty"
    first = cells[0]
    assert first["cell_type"] == "markdown", f"{notebook.name} must open with markdown"
    assert _source(first).lstrip().startswith("# "), f"{notebook.name} needs a title heading"


@pytest.mark.parametrize("notebook", ALL_NOTEBOOKS, ids=_id)
def test_every_code_cell_is_introduced(notebook: Path) -> None:
    """AGENTS.md: every code block gets a short markdown explanation before it."""
    cells = _cells(notebook)
    unintroduced = [
        index
        for index, cell in enumerate(cells)
        if cell["cell_type"] == "code"
        and (index == 0 or cells[index - 1]["cell_type"] != "markdown")
    ]
    assert not unintroduced, (
        f"{notebook.name}: code cells at {unintroduced} are not preceded by markdown"
    )


@pytest.mark.parametrize("notebook", ALL_NOTEBOOKS, ids=_id)
def test_notebook_was_executed(notebook: Path) -> None:
    """Notebooks are committed *with* their outputs, so results read on GitHub.

    A committed notebook whose cells were never run shows code and prose but no
    numbers, which defeats the point of publishing it. Regenerate with
    ``make notebooks-all``.
    """
    code_cells = [c for c in _cells(notebook) if c["cell_type"] == "code"]
    assert code_cells, f"{notebook.name} has no code cells"

    unexecuted = [
        index for index, cell in enumerate(code_cells) if cell.get("execution_count") is None
    ]
    assert not unexecuted, (
        f"{notebook.name}: code cells {unexecuted} were never executed. Run `make notebooks-all`."
    )


@pytest.mark.parametrize("notebook", ALL_NOTEBOOKS, ids=_id)
def test_notebook_was_run_top_to_bottom(notebook: Path) -> None:
    """Execution counts must be 1..N in order.

    Out-of-order counts mean the committed outputs came from cells run in some
    other sequence, so the numbers on the page need not follow from the code
    above them. That is worse than no outputs at all, because it looks fine.
    """
    counts = [
        cell.get("execution_count") for cell in _cells(notebook) if cell["cell_type"] == "code"
    ]
    assert counts == list(range(1, len(counts) + 1)), (
        f"{notebook.name}: execution counts are {counts}, expected "
        f"{list(range(1, len(counts) + 1))}. Re-run the notebook from a fresh kernel."
    )


@pytest.mark.parametrize("notebook", ANALYSIS_NOTEBOOKS, ids=_id)
def test_notebook_produced_output(notebook: Path) -> None:
    """At least some cells must have produced visible output."""
    with_output = [
        cell for cell in _cells(notebook) if cell["cell_type"] == "code" and cell.get("outputs")
    ]
    assert with_output, f"{notebook.name} produced no output at all"


@pytest.mark.parametrize("notebook", ALL_NOTEBOOKS, ids=_id)
def test_no_error_outputs(notebook: Path) -> None:
    """A committed notebook must not carry a traceback."""
    failed = [
        index
        for index, cell in enumerate(_cells(notebook))
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not failed, f"{notebook.name}: cells at {failed} committed an error output"


@pytest.mark.parametrize("notebook", ANALYSIS_NOTEBOOKS, ids=_id)
def test_required_sections_are_present(notebook: Path) -> None:
    headings = _headings(notebook)
    missing = [section for section in REQUIRED_SECTIONS if section not in headings]
    assert not missing, f"{notebook.name} is missing sections: {missing}"


@pytest.mark.parametrize("notebook", ANALYSIS_NOTEBOOKS, ids=_id)
def test_required_sections_are_ordered(notebook: Path) -> None:
    """The workflow is a sequence: you cannot diagnose a design you have not estimated."""
    headings = _headings(notebook)
    positions = [headings.index(section) for section in REQUIRED_SECTIONS if section in headings]
    assert positions == sorted(positions), (
        f"{notebook.name}: required sections appear out of order. "
        f"Expected {list(REQUIRED_SECTIONS)}, found {headings}"
    )


@pytest.mark.parametrize("notebook", ANALYSIS_NOTEBOOKS, ids=_id)
def test_results_are_interpreted(notebook: Path) -> None:
    """A printed number that nobody explains is not a finding."""
    count = _markdown_text(notebook).count(INTERPRETATION_MARKER)
    assert count >= MIN_INTERPRETATIONS, (
        f"{notebook.name} has {count} '{INTERPRETATION_MARKER}' markers, "
        f"expected at least {MIN_INTERPRETATIONS}"
    )


@pytest.mark.parametrize("notebook", ANALYSIS_NOTEBOOKS, ids=_id)
def test_estimand_is_named(notebook: Path) -> None:
    """ATE, ATT, LATE, and CATE are different quantities; the notebook must say which."""
    text = _markdown_text(notebook)
    named = [name for name in ("ATE", "ATT", "LATE", "CATE") if name in text]
    assert named, f"{notebook.name} never names its estimand (ATE, ATT, LATE, or CATE)"
