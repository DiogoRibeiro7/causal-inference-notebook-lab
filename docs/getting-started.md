# Getting started

## Install

The project uses a `src/` layout and an editable install.

=== "Linux / macOS"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
    ```

=== "Windows (PowerShell)"

    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -e ".[dev]"
    ```

Python 3.10 or newer. Add `docs` to the extras — `".[dev,docs]"` — if you intend
to build this site locally.

## Verify the install

```bash
make ci
```

That runs linting, type checking, and the test suite — the same three gates CI
enforces on every pull request. The tests are the fastest way to confirm the
estimators behave: each one checks that a known synthetic effect is recovered.

## Run the notebooks

```bash
make notebooks       # notebooks 00-02, fast
make notebooks-all   # all twelve
```

Notebooks are committed with their outputs, so a fresh clone — and the GitHub
web view — already shows every result. Re-run them when you change one:
`make notebooks-all` writes the outputs in place, and the executed file is what
you commit. CI checks that each notebook ran cleanly, in order, from a fresh
kernel.

## The benchmark dataset

Notebook 10 uses the Lalonde/NSW job training data, which is not committed.
Materialize it first:

```bash
python scripts/prepare_lalonde_job_training_dataset.py
```

This downloads the public CSV, normalizes the columns, and writes both
`data/processed/lalonde_job_training.csv` and a small manifest recording the
source, row count, and column roles. The manifest exists so that a result can be
traced back to the exact data that produced it.

!!! note "Network access required"
    The script fetches from a public GitHub mirror of the R `MatchIt` package's
    copy of the dataset. It is the only part of the project that touches the
    network, and it fails loudly rather than silently substituting data.

## Your first analysis

Every synthetic generator returns a `SyntheticDataset` carrying the data, the
true ATE, and a description. That makes it possible to ask the question that
matters: does this estimator recover an effect we already know?

```python
from causal_inference_lab import (
    aipw_ate,
    bootstrap_ate,
    ipw_ate,
    make_confounded_binary_treatment,
)

dataset = make_confounded_binary_treatment(n=4000, seed=7)
covariates = ["x1", "x2", "x3"]

point = aipw_ate(dataset.data, covariates=covariates)
interval = bootstrap_ate(
    dataset.data,
    estimator=aipw_ate,
    covariates=covariates,
    n_bootstrap_samples=200,
    seed=7,
)

print(f"true ATE: {dataset.true_ate:.3f}")
print(f"estimate: {point.estimate:.3f}")
print(f"95% CI:   [{interval.lower:.3f}, {interval.upper:.3f}]")
```

A point estimate without an interval is not a result. `bootstrap_ate` takes any
estimator with a compatible signature, so the same uncertainty machinery wraps
IPW, AIPW, and g-computation alike.

## Where the pieces live

| Path | Contents |
|---|---|
| `src/causal_inference_lab/` | Estimators, diagnostics, generators, plotting |
| `notebooks/` | Twelve analyses, each following the eight-step structure |
| `tests/` | One test module per source module |
| `scripts/` | Benchmark data preparation |
| `docs/` | This site |

## Next steps

- [Choosing an estimator](choosing-an-estimator.md) — match the method to your
  identification strategy.
- [Assumptions matrix](assumptions.md) — what each estimator needs to be valid.
- [API reference](api/index.md) — full signatures and docstrings.
