# Roadmap

This roadmap is designed to turn the repository into a strong causal inference portfolio project.

## Stage 0 — Repository foundation

**Goal:** make the project easy to run, test, and extend.

- [x] Create a notebook-first repository structure.
- [x] Add reusable source code under `src/causal_inference_lab`.
- [x] Add synthetic data generators with known causal ground truth.
- [x] Add estimators for naive difference in means, outcome regression, IPW, and AIPW.
- [x] Add balance diagnostics.
- [x] Add basic sensitivity checks.
- [x] Add unit tests for core estimators.

**Expected outcome:** anyone can clone the repo, run notebooks, and understand the project within minutes.

## Stage 1 — Core causal inference notebooks

**Goal:** show a complete causal workflow from assumptions to conclusions.

### Notebook 00 — Project overview

- Explain the business and scientific motivation.
- Introduce potential outcomes.
- Define ATE, ATT, CATE, and ITE.
- Explain why prediction alone does not answer causal questions.
- Show the repository map.

### Notebook 01 — Confounding and propensity scores

- Generate observational data with confounding.
- Compare naive difference in means against the true ATE.
- Estimate propensity scores.
- Inspect overlap / common support.
- Estimate ATE with IPW.
- Use standardized mean differences before and after weighting.

### Notebook 02 — Doubly robust AIPW

- Fit an outcome model.
- Fit a treatment model.
- Compute AIPW.
- Show why double robustness is useful.
- Add bootstrap confidence intervals.
- Compare AIPW, IPW, and g-computation.

### Notebook 03 — Heterogeneous treatment effects

- Simulate data where the treatment effect varies by covariates.
- Estimate group-level treatment effects.
- Fit a simple T-learner.
- Plot estimated CATE against the known true treatment effect.
- Discuss personalization and decision-making risk.

### Notebook 04 — Difference-in-differences

- Simulate panel data with treated and control units.
- Explain parallel trends.
- Estimate a two-way fixed effects-style model.
- Show visual pre-trends.
- Add a placebo pre-period check.

### Notebook 05 — Instrumental variables

- Simulate non-compliance and unobserved confounding.
- Introduce relevance, exclusion, and monotonicity.
- Implement two-stage least squares using `statsmodels`.
- Compare OLS and IV estimates.
- Discuss weak instruments.

### Notebook 06 — Sensitivity and reporting

- Build a causal report template.
- Include assumptions, estimand, identification strategy, estimator, diagnostics, uncertainty, and limitations.
- Add simple omitted-confounder sensitivity simulations.
- Add placebo treatment and negative-control-style checks.

**Expected outcome:** the notebooks tell a coherent causal inference story.

## Stage 2 — Real-world benchmark datasets

**Goal:** demonstrate that the workflow transfers beyond synthetic data.

Add one or more public datasets:

- Lalonde / NSW job training data.
- IHDP semi-synthetic benchmark.
- ACIC-style causal inference benchmark data.
- Marketing uplift dataset.
- Public policy difference-in-differences dataset.
- Healthcare operations synthetic benchmark.

For each dataset:

- document the causal question;
- define treatment, outcome, covariates, and unit of analysis;
- explain identification assumptions;
- compare at least three estimators;
- include diagnostics and limitations.

**Expected outcome:** the repo moves from educational examples to applied portfolio analysis.

## Stage 3 — More estimators

**Goal:** broaden the technical coverage.

Add:

- nearest-neighbour matching;
- propensity score stratification;
- entropy balancing;
- causal forests;
- R-learner;
- X-learner;
- double machine learning;
- synthetic control;
- regression discontinuity;
- interrupted time series.

Each estimator should include:

- mathematical definition;
- implementation or wrapper;
- assumptions;
- failure modes;
- synthetic-data validation;
- notebook example.

**Expected outcome:** the project becomes a compact causal inference toolkit.

## Stage 4 — Production-oriented causal analytics

**Goal:** show engineering maturity.

Add:

- configuration-driven experiment definitions;
- reproducible data generation;
- MLflow or local experiment tracking;
- HTML report generation;
- CI checks for notebooks;
- data validation with schema checks;
- Makefile targets for analysis, tests, linting, and reports.

Suggested commands:

```bash
make test
make lint
make notebooks
make report
```

**Expected outcome:** the repo demonstrates both causal reasoning and production discipline.

## Stage 5 — Decision intelligence layer

**Goal:** connect estimates to decisions.

Add decision-focused notebooks:

- treatment targeting under budget constraints;
- uplift ranking;
- net benefit estimation;
- policy learning;
- counterfactual simulation;
- fairness-aware treatment allocation.

**Expected outcome:** the repo shows how causal inference supports business and product decisions.

## Stage 6 — Publication-quality communication

**Goal:** make the project visible and easy to understand.

Add:

- article-style README;
- diagrams explaining the causal workflow;
- project website or GitHub Pages;
- short LinkedIn post;
- technical blog post;
- one-page PDF report;
- conference-style slides.

**Expected outcome:** the repo becomes strong portfolio material for senior data scientist, applied scientist, and AI product roles.

## Definition of done

The repository is mature when:

- all notebooks run from top to bottom;
- all estimators are tested on synthetic data with known effects;
- every causal claim states assumptions;
- diagnostics are included before conclusions;
- results include uncertainty;
- limitations are explicit;
- the project has a clear narrative for recruiters and technical reviewers.
