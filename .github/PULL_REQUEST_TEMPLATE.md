# Summary

<!-- What changes, and why. Link any issue this closes. -->

## Type of change

- [ ] Bug fix
- [ ] New estimator, diagnostic, or notebook
- [ ] Documentation
- [ ] Tooling, packaging, or CI
- [ ] Breaking change

## Checklist

- [ ] `make ci` passes locally
- [ ] Tests cover the change, with deterministic seeds
- [ ] Public functions have type hints and docstrings
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

## Causal correctness

<!--
Delete this section if the change does not touch an estimator, diagnostic, or
notebook conclusion. Otherwise complete it: this is the part a reviewer cannot
verify by reading the diff.
-->

- [ ] The estimand is stated explicitly (ATE, ATT, LATE, CATE)
- [ ] Identification assumptions are documented, including in `docs/assumptions.md`
- [ ] Validated on synthetic data where the true effect is known
- [ ] Compared against at least one baseline estimator
- [ ] Uncertainty is quantified, not just a point estimate
- [ ] Failure modes and limitations are stated

**True effect recovered:** <!-- e.g. "true ATE 2.0, estimate 1.97 (95% CI 1.81-2.13), n=2000, seed=42" -->

## Notes for the reviewer

<!-- Anything non-obvious: trade-offs considered, alternatives rejected, or parts you are unsure about. -->
