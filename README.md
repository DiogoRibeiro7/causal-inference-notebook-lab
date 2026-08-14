# Causal Inference Notebook Lab

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.CONCEPT_DOI.svg)](https://doi.org/10.5281/zenodo.CONCEPT_DOI)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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
│   └── 11_policy_targeting.ipynb
├── scripts/
│   └── prepare_lalonde_job_training_dataset.py
├── src/
│   └── causal_inference_lab/
├── tests/
├── data/
│   └── processed/
├── docs/
│   ├── linkedin_post.md
│   └── project_summary.md
├── README.md
├── Makefile
└── .github/workflows/ci.yml
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

## Local development

```bash
make install   # install package + dev dependencies
make lint      # formatting + linting
make typecheck # mypy checks
make test      # run unit tests
make ci        # run lint/typecheck/tests together
make notebooks # execute lightweight notebooks (00, 01, 02)
```

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

Each notebook states:

- causal question
- treatment, outcome, covariates, and unit of analysis
- estimand
- assumptions
- effect estimates
- diagnostics
- uncertainty
- limitations

This repository is intentionally explicit about uncertainty and assumptions to avoid overclaiming causal results.

## Citation and archiving

Releases are archived on Zenodo and receive a DOI. Deposit metadata lives in
[.zenodo.json](.zenodo.json); human/GitHub-facing citation metadata lives in
[CITATION.cff](CITATION.cff). Keep `version` in sync across `CITATION.cff`,
`.zenodo.json`, and `pyproject.toml` when tagging a release.

Cite the specific version and commit used for reproduction:

```bibtex
@software{ribeiro_causal_inference_notebook_lab,
  author  = {Ribeiro, Diogo},
  title   = {Causal Inference Notebook Lab},
  version = {0.1.0},
  doi     = {10.5281/zenodo.CONCEPT_DOI},
  url     = {https://github.com/DiogoRibeiro7/causal-inference-notebook-lab}
}
```

### Enabling the Zenodo archive (one-time)

1. Sign in at [zenodo.org](https://zenodo.org) with the GitHub account, then open
   **Account → GitHub** and flip the switch on `DiogoRibeiro7/causal-inference-notebook-lab`.
2. Push a tag and publish a GitHub release (for example `v0.1.0`). Zenodo picks up the
   webhook, mints a DOI, and reads `.zenodo.json` for the deposit metadata.
3. Zenodo issues two DOIs: a **concept DOI** that always resolves to the latest version,
   and a **version DOI** for that specific release. Replace every `CONCEPT_DOI` placeholder
   above (badge, badge link, and BibTeX) with the concept DOI suffix.
4. Later releases inherit the metadata automatically; only the version fields need bumping.

The badge stays broken until step 3 is done, which is the intended signal that the first
release has not been archived yet.
