# Causal Inference Notebook Lab

[![CI](https://github.com/DiogoRibeiro7/causal-inference-notebook-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/causal-inference-notebook-lab/actions/workflows/ci.yml)
[![Docs](https://github.com/DiogoRibeiro7/causal-inference-notebook-lab/actions/workflows/docs.yml/badge.svg)](https://diogoribeiro7.github.io/causal-inference-notebook-lab/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21936500.svg)](https://doi.org/10.5281/zenodo.21936500)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**[Documentation](https://diogoribeiro7.github.io/causal-inference-notebook-lab/)** ·
[Getting started](https://diogoribeiro7.github.io/causal-inference-notebook-lab/getting-started/) ·
[Assumptions matrix](https://diogoribeiro7.github.io/causal-inference-notebook-lab/assumptions/) ·
[API reference](https://diogoribeiro7.github.io/causal-inference-notebook-lab/api/)

A notebook-first causal inference portfolio focused on transparent causal workflows.

The project is structured to separate reusable estimators from analysis notebooks:

- Synthetic datasets with known ground truth for reproducible development
- Reusable estimator implementations in `src/`
- Notebook-driven examples in `notebooks/`
- Automated checks through Makefile and CI

The workflow is intentional:

1. define the causal question;
2. define treatment, outcome, covariates, and unit of analysis;
3. define the estimand;
4. state identification assumptions;
5. estimate with multiple methods;
6. diagnose validity and uncertainty;
7. communicate results with limitations and recommendation.

```mermaid
flowchart LR
  Q[Question] --> D[Data]
  D --> E[Estimation]
  E --> B[Balance and diagnostics]
  B --> U[Uncertainty + sensitivity]
  U --> R[Causal report]
  R --> P[Decision support]
  P --> M[Production caveats]
```

## Repository structure

```text
causal-inference-notebook-lab/
├── notebooks/
│   ├── 00_project_overview.ipynb
│   ├── 01_confounding_propensity_scores.ipynb
│   ├── 02_doubly_robust_aipw.ipynb
│   ├── 03_heterogeneous_treatment_effects.ipynb
│   ├── 04_difference_in_differences.ipynb
│   ├── 05_instrumental_variables.ipynb
│   ├── 06_sensitivity_and_reporting.ipynb
│   ├── 07_double_machine_learning.ipynb
│   ├── 08_regression_discontinuity.ipynb
│   ├── 09_synthetic_control.ipynb
│   ├── 10_lalonde_job_training.ipynb
│   ├── 11_policy_targeting.ipynb
│   └── 12_ihdp_benchmark.ipynb
├── scripts/
│   ├── prepare_lalonde_job_training_dataset.py
│   ├── prepare_ihdp_dataset.py
│   └── build_docs_notebooks.py
├── src/
│   └── causal_inference_lab/       # estimators, diagnostics, generators
├── tests/                          # one module per source module
├── data/
│   └── processed/                  # generated, not committed
├── docs/                           # MkDocs site source
│   ├── index.md
│   ├── getting-started.md
│   ├── choosing-an-estimator.md
│   ├── assumptions.md
│   ├── api/
│   └── about/
├── .github/workflows/
│   ├── ci.yml                      # lint, typecheck, tests, fast notebooks
│   ├── docs.yml                    # build and deploy to GitHub Pages
│   └── notebooks-nightly.yml       # execute all thirteen, nightly
├── CONTRIBUTING.md
├── CHANGELOG.md
├── mkdocs.yml
├── Makefile
└── README.md
```

## Setup

The project uses a standard `src/` layout and editable installs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Install the `docs` extra as well — `pip install -e ".[dev,docs]"` — to build the
documentation site locally.

## Local development

```bash
make install        # package + dev dependencies
make lint           # formatting + linting
make typecheck      # mypy checks
make test           # unit tests
make coverage       # tests with branch coverage, enforcing the 80% floor
make ci             # lint, typecheck, and tests together
make notebooks      # execute the lightweight notebooks (00, 01, 02)
make notebooks-all  # execute all thirteen
make docs           # serve the documentation site locally
```

`pre-commit install` wires ruff, mypy, and file hygiene checks into your
commits. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow and the
definition of done for a new estimator.

Notebooks are committed **with their outputs**, so every result is readable
directly on GitHub — no clone, no kernel. Re-run with `make notebooks-all` after
changing one; CI verifies each was executed cleanly and in order.

## Benchmark data preparation

Run the benchmark script to materialize the Lalonde/NSW dataset locally before notebook 10:

```bash
python scripts/prepare_lalonde_job_training_dataset.py
```

This saves `data/processed/lalonde_job_training.csv` and writes a small processing manifest for reproducibility.

## Portfolio notes

### What I would do in production

- Add explicit data schemas and validation checks before estimators run.
- Pin dependency versions and lock model/feature configs.
- Track estimation inputs and outputs (seeds, assumptions, diagnostics, uncertainty).
- Add drift and overlap monitoring so re-runs fail fast when identifying assumptions are likely broken.
- Add approval gates: diagnostics thresholds, placebo checks, CI notebook execution.

### Known limitations

- Some notebooks use simplified estimators intentionally for educational clarity.
- Heavyweight uncertainty methods are present only for selected estimands.
- Synthetic examples are not replacements for messy real-world data with measurement error and missingness.
- No production deployment scaffolding (feature store, serving layer, or automated monitoring) is implemented yet.

## Methods covered

- naive difference in means
- propensity / logistic adjustment
- inverse probability weighting
- outcome regression and AIPW
- matching (propensity and nearest-neighbor)
- doubly robust causal estimators (DML)
- regression discontinuity
- synthetic control
- heterogeneous effect estimators (S/T/X)
- bootstrap uncertainty
- sensitivity and reporting
- benchmark-oriented policy targeting

## Causal workflow discipline in each notebook

Every analysis notebook carries the same eight sections, in order:

1. causal question
2. data and design — treatment, outcome, covariates, unit of analysis
3. estimand
4. identification assumptions
5. estimation
6. diagnostics
7. uncertainty
8. limitations

This is enforced, not merely intended. [`tests/test_notebook_structure.py`](tests/test_notebook_structure.py)
checks the sections are present and ordered, that every code cell is introduced
by prose, that the estimand is named explicitly, and that results are
interpreted rather than left as bare output. A notebook that drops its
assumptions section fails CI.

The tests check presence, not honesty — no test can tell you an assumptions
section is candid. What they prevent is the silent drift that had left eight of
the twelve notebooks below this standard before v0.3.0.

The repository is deliberately explicit about uncertainty and assumptions to
avoid overclaiming causal results. [Notebook 10](notebooks/10_lalonde_job_training.ipynb)
is the clearest example: it applies every estimator here to the Lalonde/NSW
benchmark and concludes that none of them supports a causal claim.

## Citation and archiving

Releases are archived on Zenodo. Deposit metadata lives in [.zenodo.json](.zenodo.json);
human/GitHub-facing citation metadata lives in [CITATION.cff](CITATION.cff).

Two DOIs exist. The **concept DOI** always resolves to the newest release and is the one
to cite when the exact version does not matter:

- Concept DOI: [10.5281/zenodo.21936500](https://doi.org/10.5281/zenodo.21936500)
- v0.3.0: [10.5281/zenodo.21959303](https://doi.org/10.5281/zenodo.21959303)
- v0.2.0: [10.5281/zenodo.21939879](https://doi.org/10.5281/zenodo.21939879)
- v0.1.0: [10.5281/zenodo.21936501](https://doi.org/10.5281/zenodo.21936501)

For reproduction, cite the specific version and commit used:

```bibtex
@software{ribeiro_causal_inference_notebook_lab,
  author  = {Ribeiro, Diogo},
  title   = {Causal Inference Notebook Lab},
  version = {0.3.0},
  doi     = {10.5281/zenodo.21959303},
  url     = {https://github.com/DiogoRibeiro7/causal-inference-notebook-lab}
}
```

### Cutting a new release

1. Bump `version` in `pyproject.toml`, `CITATION.cff`, and `.zenodo.json` together.
2. Tag and publish a GitHub release. The Zenodo webhook fires on the `released` event,
   mints a version DOI, and re-reads `.zenodo.json` for the deposit metadata.
3. Update the version DOI in the list above and in the BibTeX block. The badge tracks the
   concept DOI and does not change.
