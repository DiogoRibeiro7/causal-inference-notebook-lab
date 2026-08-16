# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each released version is archived on Zenodo with its own DOI; see the
[citation section](https://github.com/DiogoRibeiro7/causal-inference-notebook-lab#citation-and-archiving).

## [Unreleased]

Nothing yet.

## [0.4.0] - 2026-08-16

Two additions, both closing gaps the roadmap had open: a second benchmark
dataset, and the fairness analysis notebook 11 had promised and not delivered.

Additive only — no existing signature or behaviour changes.

### Added

- **IHDP semi-synthetic benchmark** (notebook 12) and
  `scripts/prepare_ihdp_dataset.py`, which downloads ten replications with a
  processing manifest marking the ground-truth columns an estimator may not
  read.

  IHDP complements Lalonde by having known individual effects, so estimators can
  be scored rather than compared. The result is uncomfortable and is the point of
  the notebook: **the unadjusted difference in means is the most accurate
  estimator on this benchmark**, beating AIPW in 8 of 10 replications, in a
  setting where conditional ignorability holds by construction. Adjustment is
  not free, and nothing in an analysis's output reveals whether it helped.

  The notebook also shows a single pathological replication reordering the
  method ranking depending on whether you report the mean or the median, and the
  linear S-learner again reporting a CATE standard deviation of exactly 0.000 —
  this time on real covariates.
- `tests/test_prepare_ihdp.py`, pinning the positional column order of the
  published CSVs. Mislabelling it would produce a dataset that loads cleanly and
  means something different.
- **Fairness-aware targeting** (notebook 13), `causal_inference_lab.fairness`,
  and `make_service_allocation_population`. Notebook 11 stated that targeting
  "can amplify bias" and left it there; this measures it.

  `allocation_table` and `allocation_disparity` report what share of each group
  a policy treats; `parity_constrained_selection` builds an equal-rate
  alternative so the trade-off can be priced. With identical true benefit across
  groups, the benefit-ranked policy still allocates unevenly — 18.5% against
  23.4% — and equalising it costs nothing. Where benefit genuinely differs,
  parity costs 2.3% of total gain, though a bootstrap shows that figure is not
  distinguishable from zero on this sample.

### Changed

- `make data` now prepares both benchmarks.
- `ROADMAP.md` rewritten to reflect actual status. The "definition of done" is
  met as of v0.3.0, and several of its criteria are now enforced by tests rather
  than asserted.

## [0.3.0] - 2026-08-15

This version is [10.5281/zenodo.21959303](https://doi.org/10.5281/zenodo.21959303).

The notebooks release. Every analysis was rewritten to the workflow the
repository had always documented but never enforced, and three bugs surfaced in
the process — two of them in the library, found by trying to write analyses that
used it.

**Behaviour changes worth noting before upgrading:** `instrumental_variables_ate`
now reports `estimand="LATE"` rather than `"ATE"`, so code branching on that
string behaves differently; and calls that previously raised `AttributeError`
when passed an ensemble model now succeed.

### Added

- `tests/test_notebook_structure.py`, making the documented eight-step workflow
  a build-time contract: sections present and ordered, every code cell
  introduced by prose, the estimand named, results interpreted, and the
  notebook executed cleanly from a fresh kernel.
- `tests/test_custom_models.py`, covering every estimator entry point that
  accepts a caller-supplied model.

### Changed

- All twelve notebooks rewritten to the workflow standard. Eight had drifted
  below it and three were a single sentence and a `print`. Each now compares
  against a baseline, runs diagnostics, quantifies uncertainty, and states what
  it cannot conclude.
- **Notebooks are now committed with their outputs**, so results are readable on
  GitHub without cloning. The `nbstripout` pre-commit hook is removed, and the
  structure tests instead assert every cell was executed, that execution counts
  run 1..N from a fresh kernel, and that no traceback was committed. All twelve
  come to 248 KB, since the outputs are text tables rather than figures.
- The documentation build copies the committed outputs rather than re-executing
  the notebooks, which removes a network dependency and cuts the docs job from
  roughly four minutes to seconds. `--execute` still forces a re-run.

### Fixed

- Estimators accepting a caller-supplied model crashed when given an unfitted
  scikit-learn ensemble. Ten call sites used `model or Default()`, which
  evaluates `bool(model)`; an unfitted ensemble raises `AttributeError` from
  `__len__`. This broke the main reason to supply a model at all — non-linear
  nuisance functions in double machine learning and meta-learners.
- `instrumental_variables_ate` reported `estimand="ATE"`. Two-stage least
  squares identifies the LATE, and the two coincide only under homogeneous
  effects — the assumption an instrument is usually invoked to avoid.
- Notebook 11's net-benefit calculation added `true_ite` to outcomes that
  already contained the effect for treated units, then returned the sum of two
  group means — a quantity that moves with group size and measures nothing.

## [0.2.0] - 2026-08-14

Repository maturity release. No estimator behaviour changed; everything here is
scaffolding, tooling, and documentation around the existing science. This
version is [10.5281/zenodo.21939879](https://doi.org/10.5281/zenodo.21939879).

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

[Unreleased]: https://github.com/DiogoRibeiro7/causal-inference-notebook-lab/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/DiogoRibeiro7/causal-inference-notebook-lab/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/DiogoRibeiro7/causal-inference-notebook-lab/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/DiogoRibeiro7/causal-inference-notebook-lab/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DiogoRibeiro7/causal-inference-notebook-lab/releases/tag/v0.1.0
