# Contributing

Thanks for your interest in this project. It is a causal inference lab: reusable
estimators under `src/causal_inference_lab/`, and notebooks that apply them to
synthetic and benchmark data.

The bar for contributions is not "the code runs". It is "the causal claim is
defensible, and the assumptions behind it are written down".

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev,docs]"
pre-commit install
```

## Everyday commands

| Command | What it does |
|---|---|
| `make lint` | `ruff check` plus format verification |
| `make format` | Apply formatting fixes |
| `make typecheck` | `mypy src` |
| `make test` | Run the test suite |
| `make coverage` | Tests with a branch-coverage report and the 80% gate |
| `make ci` | Lint, typecheck, and test together — what CI runs |
| `make notebooks` | Execute the three lightweight notebooks in place |
| `make notebooks-all` | Execute all thirteen, including both benchmarks |
| `make docs` | Serve the documentation site locally |
| `make docs-build` | Build the site strictly, failing on warnings |

`pre-commit install` is worth the ten seconds: it runs ruff, mypy, and file
hygiene checks before each commit, so CI rarely tells you something you could
have learned locally.

## Standards

Repository-wide engineering and scientific conventions live in
[AGENTS.md](https://github.com/DiogoRibeiro7/causal-inference-notebook-lab/blob/main/AGENTS.md)
— type hints on public functions, docstrings, validated
inputs, clear `ValueError` messages, deterministic seeds in tests, and the
eight-step causal structure every notebook follows. Read it before your first
substantive change.

A few points worth repeating:

- **Notebooks are committed *with* their outputs**, so results are readable on
  GitHub without cloning and running anything. After changing a notebook, run
  `make notebooks-all` and commit the executed file. The structure tests check
  every cell ran, that execution counts are 1..N from a fresh kernel, and that
  no traceback was committed — stale or out-of-order outputs fail CI.
- **Do not add heavyweight dependencies** without justification in the PR.
- **Plotting stays separate from estimation.** Estimators return numbers and
  result objects; `plotting.py` turns them into figures.

## Adding an estimator

A new estimator is not done when it produces a number. It is done when it has:

1. An implementation under `src/causal_inference_lab/`, with validated inputs
   and a frozen result dataclass.
2. A synthetic-data test where the true effect is known, asserting the estimate
   recovers it within a stated tolerance.
3. A notebook example following the eight-step structure.
4. A short mathematical statement of what it estimates.
5. Its identification assumptions and failure modes, added to
   the [assumptions matrix](https://diogoribeiro7.github.io/causal-inference-notebook-lab/assumptions/)
   (`docs/assumptions.md`).
6. A comparison against at least one baseline estimator.
7. An export in `__init__.py` and an API page under `docs/api/`.

## Pull requests

Branch from `main`, keep commits focused, and write commit messages that explain
*why* rather than restating the diff.

`main` is protected: CI must pass on Python 3.10, 3.11, and 3.12, and every PR
needs a review. Fill in the PR template — particularly the causal-correctness
checklist when the change touches an estimator, because that is the part
reviewers cannot verify by reading the diff alone.

## Reporting problems

Bugs and feature requests go through the [issue
templates](https://github.com/DiogoRibeiro7/causal-inference-notebook-lab/issues/new/choose).
Security issues follow
[SECURITY.md](https://github.com/DiogoRibeiro7/causal-inference-notebook-lab/blob/main/SECURITY.md)
instead — please do not open a
public issue for those.

A statistical bug — an estimator that returns a biased or wrong estimate under
stated assumptions — is more serious than a crash. If you find one, say so
prominently in the issue title, and include the seed and data-generating process
that reproduces it.
