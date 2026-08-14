# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each released version is archived on Zenodo with its own DOI; see the
[citation section](https://github.com/DiogoRibeiro7/causal-inference-notebook-lab#citation-and-archiving).

## [Unreleased]

Nothing yet.

## [0.2.0] - 2026-08-14

Repository maturity release. No estimator behaviour changed; everything here is
scaffolding, tooling, and documentation around the existing science.

### Added

- Contributor scaffolding: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, this changelog, issue forms, and a pull request template.
- Documentation site built with MkDocs Material, including an API reference
  generated from docstrings, a per-estimator assumptions matrix, and the
  notebooks rendered as pages.
- `pre-commit` configuration running ruff, ruff-format, mypy, `nbstripout`, and
  file hygiene hooks.
- `py.typed` marker, so downstream projects consume the package's type hints.
- `causal_inference_lab.__version__`, now the single source of the version;
  `pyproject.toml` reads it dynamically.
- `tests/test_metadata.py`, asserting the version agrees across the package,
  the installed distribution, `CITATION.cff`, and `.zenodo.json`.
- Dependabot for GitHub Actions and pip updates.
- Nightly workflow executing all twelve notebooks and opening an issue on
  failure.
- Branch-coverage reporting with an 80% floor enforced in CI. Branch coverage
  counts partial branches as misses, so 80% here is stricter than the 85% the
  same suite scores line-only.

### Changed

- Development dependencies now carry upper bounds. Unpinned tooling is what
  broke CI silently between June and August 2026.
- CI bumped to `actions/checkout@v5` and `actions/setup-python@v6` with pip
  caching, plus concurrency groups so superseded runs are cancelled.
- Documentation is published to GitHub Pages from `main`.

### Fixed

- The issue chooser linked to GitHub Discussions, which is not enabled on this
  repository.

## [0.1.0] - 2026-08-14

First archived release. Concept DOI
[10.5281/zenodo.21936500](https://doi.org/10.5281/zenodo.21936500); this version
is [10.5281/zenodo.21936501](https://doi.org/10.5281/zenodo.21936501).

### Added

- Reusable estimators under `src/causal_inference_lab/`: difference in means,
  g-computation, IPW, AIPW, propensity and nearest-neighbour matching, double
  machine learning, difference-in-differences, instrumental variables,
  regression discontinuity, synthetic control, S/T/X meta-learners, bootstrap
  uncertainty, balance diagnostics, sensitivity analysis, and reporting.
- Synthetic data generators exposing the true treatment effect.
- Twelve notebooks, each stating causal question, estimand, identification
  assumptions, diagnostics, uncertainty, and limitations.
- `scripts/prepare_lalonde_job_training_dataset.py`, materializing the
  Lalonde/NSW benchmark with a processing manifest.
- Zenodo archiving metadata and `CITATION.cff`.
- MIT license.
- CI running lint, typecheck, tests, and lightweight notebook execution on
  Python 3.10, 3.11, and 3.12.

### Fixed

- Synthetic control panel indexing, which produced 34 mypy errors under current
  pandas-stubs releases.
- The mypy `python_version` pin, which made the 3.12 CI leg fail inside the
  numpy stubs before reaching project code.

[Unreleased]: https://github.com/DiogoRibeiro7/causal-inference-notebook-lab/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/DiogoRibeiro7/causal-inference-notebook-lab/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DiogoRibeiro7/causal-inference-notebook-lab/releases/tag/v0.1.0
