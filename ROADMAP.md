# Roadmap

This roadmap tracks turning the repository into a strong causal inference
portfolio project.

**Status at v0.3.0:** Stages 0, 1 and 6 are complete, Stage 5 substantially so.
The "definition of done" at the bottom is met — and is now enforced by tests
rather than asserted in prose. What remains is breadth (more estimators, more
benchmark datasets) and the production layer of Stage 4.

## Stage 0 — Repository foundation

**Goal:** make the project easy to run, test, and extend.

- [x] Create a notebook-first repository structure.
- [x] Add reusable source code under `src/causal_inference_lab`.
- [x] Add synthetic data generators with known causal ground truth.
- [x] Add estimators for naive difference in means, outcome regression, IPW, and AIPW.
- [x] Add balance diagnostics.
- [x] Add basic sensitivity checks.
- [x] Add unit tests for core estimators.

Delivered beyond the original plan:

- [x] MIT license, `CITATION.cff`, and Zenodo archiving with a concept DOI.
- [x] Contributor scaffolding: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
      `SECURITY.md`, `CHANGELOG.md`, issue forms, and a PR template carrying a
      causal-correctness checklist.
- [x] `pre-commit`, `py.typed`, single-sourced `__version__`, and upper-bounded
      dev dependencies with grouped Dependabot updates.
- [x] Branch coverage with an 80% floor, enforced in CI.

**Outcome:** met. A clone runs `make ci` and gets lint, types, and 155 tests.

## Stage 1 — Core causal inference notebooks

**Goal:** show a complete causal workflow from assumptions to conclusions.

- [x] Notebook 00 — project overview and potential-outcomes vocabulary.
- [x] Notebook 01 — confounding, propensity scores, overlap, balance, IPW.
- [x] Notebook 02 — doubly robust AIPW, with double robustness demonstrated by
      misspecifying each nuisance model in turn.
- [x] Notebook 03 — heterogeneous effects and the S/T/X learners.
- [x] Notebook 04 — difference-in-differences with pre-trend and placebo checks.
- [x] Notebook 05 — instrumental variables, weak instruments, and the LATE.
- [x] Notebook 06 — sensitivity analysis and the structured causal report.

All thirteen notebooks follow the same eight-section structure, are committed
with their outputs, and are checked by `tests/test_notebook_structure.py`.

**Outcome:** met.

## Stage 2 — Real-world benchmark datasets

**Goal:** demonstrate that the workflow transfers beyond synthetic data.

- [x] Lalonde / NSW job training data (notebook 10), with a processing manifest.
- [x] IHDP semi-synthetic benchmark (notebook 12), ten replications.
- [ ] ACIC-style causal inference benchmark data.
- [ ] Marketing uplift dataset.
- [ ] Public policy difference-in-differences dataset.
- [ ] Healthcare operations synthetic benchmark.

The Lalonde notebook is arguably the most valuable in the repository precisely
because its conclusion is negative: no estimator here recovers the experimental
benchmark, and the notebook says so.

**Outcome:** partially met — two benchmarks, each thoroughly done. Lalonde
shows every estimator failing; IHDP shows the unadjusted estimator winning.

## Stage 3 — More estimators

**Goal:** broaden the technical coverage.

- [x] nearest-neighbour matching
- [x] propensity score matching
- [x] X-learner (and S- and T-learners)
- [x] double machine learning
- [x] synthetic control
- [x] regression discontinuity
- [ ] propensity score stratification
- [ ] entropy balancing
- [ ] causal forests
- [ ] R-learner
- [ ] interrupted time series

Each new estimator must meet the definition of done in
[CONTRIBUTING.md](CONTRIBUTING.md): implementation, a synthetic-data test
against a known effect, a notebook example, its assumptions and failure modes
added to the assumptions matrix, and a comparison against a baseline.

**Outcome:** six of eleven.

## Stage 4 — Production-oriented causal analytics

**Goal:** show engineering maturity.

- [x] reproducible data generation (seeded generators, benchmark manifest)
- [x] CI checks for notebooks (fast set per PR, all of them nightly)
- [x] Makefile targets for analysis, tests, linting, and docs
- [ ] configuration-driven experiment definitions
- [ ] experiment tracking (MLflow or a local equivalent)
- [ ] HTML report generation — `CausalReport.to_markdown()` exists;
      `reports/` is still empty and there is no `make report`
- [ ] data validation with schema checks — estimators validate their inputs
      inline, but there is no schema layer

**Outcome:** the CI and reproducibility half is done; the experiment-management
half is not.

## Stage 5 — Decision intelligence layer

**Goal:** connect estimates to decisions.

- [x] treatment targeting under budget constraints (notebook 11)
- [x] uplift ranking (notebook 11)
- [x] net benefit estimation (notebook 11, including the cost at which
      universal treatment destroys value while targeted treatment does not)
- [ ] policy learning
- [ ] counterfactual simulation
- [x] fairness-aware treatment allocation (notebook 13), with allocation
      auditing and parity-constrained targeting in `fairness.py`

**Outcome:** substantially met — policy learning and counterfactual simulation
remain.

## Stage 6 — Publication-quality communication

**Goal:** make the project visible and easy to understand.

- [x] article-style README
- [x] diagrams explaining the causal workflow
- [x] project website (MkDocs Material on GitHub Pages, with an API reference,
      an assumptions matrix, and the notebooks rendered with their outputs)
- [x] short LinkedIn post (draft in `docs/about/`)
- [ ] technical blog post
- [ ] one-page PDF report
- [ ] conference-style slides

**Outcome:** met for the web-facing material.

## Definition of done

The repository is mature when:

- [x] all notebooks run from top to bottom — executed nightly, and their
      committed outputs are checked for clean 1..N execution
- [x] all estimators are tested on synthetic data with known effects
- [x] every causal claim states assumptions — enforced by the structure tests
- [x] diagnostics are included before conclusions — section ordering is enforced
- [x] results include uncertainty
- [x] limitations are explicit
- [x] the project has a clear narrative for recruiters and technical reviewers

This is met as of v0.3.0. The tests check that these sections *exist*, not that
they are candid; that part still depends on the person writing them.
