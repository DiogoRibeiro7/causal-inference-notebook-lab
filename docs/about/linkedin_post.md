# Draft LinkedIn post

I finished a causal inference portfolio focused on transparent, assumption-aware workflows.

I built `causal-inference-notebook-lab` around three things that matter in applied causal work:

- Define the question and estimand before fitting models.
- Make identification assumptions explicit (unconfoundedness, overlap, SUTVA, etc.).
- Compare methods against diagnostics, uncertainty, and sensitivity checks.

The project includes synthetic data generators with known ground truth, reusable estimators in `src/`, and notebook examples covering matching, AIPW, DML, RDD, synthetic control, and policy targeting.

It also has production-minded wiring: lint/typecheck/tests, lightweight notebook execution in CI, and a reproducible public benchmark preparation script for Lalonde / NSW.

The repo is intentionally explicit that assumptions, not just p-values, drive causal claims. That is the difference between a model and a causal analysis.
