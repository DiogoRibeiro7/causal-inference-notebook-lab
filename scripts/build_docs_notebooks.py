"""Stage the notebooks into the documentation tree.

Notebooks are committed with their outputs, so by default this just copies them
and the site renders exactly what is in the repository — the same numbers a
reader sees on GitHub. That keeps the documentation build fast and free of any
network dependency.

Pass ``--execute`` to re-run them first. Execution uses ``notebooks/`` as the
kernel's working directory, because each notebook locates the project with::

    PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

Executing a *copy* under ``docs/notebooks`` would satisfy that check while
resolving the root to ``docs/``, where ``src`` and ``scripts`` do not exist. So
we read from ``notebooks/``, execute with that directory as the cwd, and write
the result into the docs tree, leaving the sources untouched.

Notebook 10 needs the Lalonde benchmark, downloaded by
``scripts/prepare_lalonde_job_training_dataset.py``. Run that first when
executing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "notebooks"
DESTINATION = REPO_ROOT / "docs" / "notebooks"
DEFAULT_TIMEOUT = 900


def build(
    source: Path = SOURCE,
    destination: Path = DESTINATION,
    timeout: int = DEFAULT_TIMEOUT,
    execute: bool = False,
) -> list[Path]:
    """Stage each notebook into the docs tree, optionally re-executing it.

    Args:
        source: Directory holding the canonical notebooks.
        destination: Directory inside the MkDocs tree to write into.
        timeout: Per-cell execution timeout in seconds.
        execute: When True, re-run the notebooks instead of copying their
            committed outputs.

    Returns:
        The notebook paths written, sorted by name.

    Raises:
        FileNotFoundError: If the source directory does not exist.
    """
    if not source.is_dir():
        raise FileNotFoundError(f"No notebook directory at {source}")

    # Rebuild from scratch so a notebook deleted upstream does not linger.
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    notebooks = sorted(source.glob("*.ipynb"))
    if not execute:
        for notebook in notebooks:
            shutil.copy2(notebook, destination / notebook.name)
        return [destination / notebook.name for notebook in notebooks]

    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    # Never try to open a GUI window from a headless build.
    os.environ.setdefault("MPLBACKEND", "Agg")

    written = []
    for index, notebook_path in enumerate(notebooks, start=1):
        print(f"[{index}/{len(notebooks)}] executing {notebook_path.name}", flush=True)
        notebook = nbformat.read(notebook_path, as_version=4)
        processor = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
        # The kernel's cwd is `notebooks/`, matching how the notebooks are run
        # by `make notebooks` and how they resolve PROJECT_ROOT.
        processor.preprocess(notebook, {"metadata": {"path": str(source)}})

        target = destination / notebook_path.name
        nbformat.write(notebook, target)
        written.append(target)

    return written


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command line arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Re-run the notebooks instead of copying their committed outputs.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-cell timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    args = parser.parse_args(argv)

    written = build(timeout=args.timeout, execute=args.execute)
    verb = "Executed" if args.execute else "Copied"
    print(f"{verb} {len(written)} notebooks into {DESTINATION.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
