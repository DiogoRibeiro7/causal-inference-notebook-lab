# Project summary (one pager)

## What this repo demonstrates

This repository is a practical causal-inference portfolio that combines:

- robust scientific workflow (questions, assumptions, estimands)
- reusable estimation code in `src/`
- notebook-driven communication
- benchmark-oriented examples and reproducibility gates

## Covered methods

- Confounding controls (naive, IPW, outcome regression, AIPW)
- Matching (propensity score matching, nearest-neighbor matching)
- Doubly robust/orthogonal learning (DML)
- Regression discontinuity
- Synthetic control for one treated unit
- Heterogeneous effects (S/T/X learners)
- Bootstrap uncertainty and sensitivity checks
- Policy targeting notebook with budget constraints
- Public benchmark workflow for Lalonde / NSW job training data

## Design choices

- Deterministic seeds in synthetic generators and tests
- No global mutable state in estimator code
- Explicit separation of estimation and plotting
- Markdown sections in notebooks to force interpretation steps

## Production readiness highlights

- `Makefile` automates linting, type-checking, tests, and lightweight notebook execution
- CI executes quality gates on Python 3.10/3.11/3.12
- Public benchmark preparation script stores a processed artifact and manifest
- Reporting object supports a structured, assumption-forward causal report

## What I would ship next

- Add cross-validated nuisance model selection for DML
- Add dedicated feature schemas and automatic input validation in loaders
- Add deployment checklist and experiment tracking for reproducibility at scale

The intent is to show not only that methods exist, but that they are used in a credible analysis sequence.
