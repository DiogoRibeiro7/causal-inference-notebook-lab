# Causal Inference Notebook Lab

A notebook-first causal inference lab. Reusable estimators live in
`causal_inference_lab`; the notebooks apply them to synthetic data with known
ground truth and to the Lalonde/NSW benchmark.

The organising idea is that a causal estimate is only as good as the assumptions
you are willing to defend. Every analysis notebook states them, every estimator
documents the conditions under which it identifies an effect, and results are
reported with uncertainty and limitations rather than a single confident number.

That structure is enforced by tests rather than left to good intentions: the
eight sections below must be present and ordered, the estimand must be named,
and every result must be interpreted, or CI fails.

!!! warning "This is a lab, not a production causal engine"
    Some estimators are deliberately simplified for clarity. Synthetic examples
    are not a substitute for messy real data with measurement error and
    missingness. Read the [assumptions matrix](assumptions.md) before applying
    anything here to a decision that matters.

## The workflow

Every analysis follows the same seven steps, in order:

```mermaid
flowchart LR
  Q[Question] --> D[Data]
  D --> E[Estimation]
  E --> B[Balance and diagnostics]
  B --> U[Uncertainty + sensitivity]
  U --> R[Causal report]
  R --> P[Decision support]
```

1. Define the causal question.
2. Define treatment, outcome, covariates, and unit of analysis.
3. Define the estimand — ATE, ATT, LATE, or CATE.
4. State identification assumptions.
5. Estimate with multiple methods.
6. Diagnose validity and quantify uncertainty.
7. Communicate results with limitations and a recommendation.

The order matters. Choosing an estimator before defining the estimand is how
analyses end up answering a question nobody asked.

## Quick example

```python
from causal_inference_lab import (
    aipw_ate,
    difference_in_means,
    ipw_ate,
    make_confounded_binary_treatment,
)

dataset = make_confounded_binary_treatment(n=2000, seed=42)
covariates = ["x1", "x2", "x3"]

print(f"true ATE:  {dataset.true_ate:.3f}")
print(f"naive:     {difference_in_means(dataset.data).estimate:.3f}")
print(f"IPW:       {ipw_ate(dataset.data, covariates=covariates).estimate:.3f}")
print(f"AIPW:      {aipw_ate(dataset.data, covariates=covariates).estimate:.3f}")
```

```text
true ATE:  1.972
naive:     3.818
IPW:       1.899
AIPW:      2.050
```

The generator exposes `true_ate`, so you can check whether an estimator recovers
what it claims to. That check is the point of the synthetic datasets — and the
gap between the naive and adjusted estimates is the confounding this dataset was
built to contain.

## Where to go next

<div class="grid cards" markdown>

- :material-rocket-launch: **[Getting started](getting-started.md)**

    Install, run the tests, execute the notebooks.

- :material-sitemap: **[Choosing an estimator](choosing-an-estimator.md)**

    Which method fits your identification strategy.

- :material-table-check: **[Assumptions matrix](assumptions.md)**

    What each estimator requires, and how it fails.

- :material-api: **[API reference](api/index.md)**

    Generated from the source docstrings.

</div>

## Citing

Archived on Zenodo. Cite the concept DOI
[10.5281/zenodo.21936500](https://doi.org/10.5281/zenodo.21936500) for the
project, or a version DOI for reproduction. See
[`CITATION.cff`](https://github.com/DiogoRibeiro7/causal-inference-notebook-lab/blob/main/CITATION.cff).
