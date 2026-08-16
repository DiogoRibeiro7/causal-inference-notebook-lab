# API reference

Generated from the source docstrings. Everything listed here is exported from
the top-level package:

```python
from causal_inference_lab import aipw_ate, make_confounded_binary_treatment
```

The package ships a `py.typed` marker, so type checkers in downstream projects
use these annotations rather than falling back to `Any`.

## By task

| I want to... | Module |
|---|---|
| Generate data with a known effect | [Data generators](data_generators.md) |
| Estimate an ATE under ignorability | [Estimators](estimators.md) |
| Estimate with many covariates | [Double machine learning](dml.md) |
| Build an interpretable matched sample | [Matching](matching.md) |
| Estimate effects that vary by unit | [Meta-learners](meta_learners.md) |
| Use panel data and parallel trends | [Difference-in-differences](difference_in_differences.md) |
| Use an instrument | [Instrumental variables](instrumental_variables.md) |
| Exploit a threshold rule | [Regression discontinuity](rdd.md) |
| Handle one treated unit | [Synthetic control](synthetic_control.md) |
| Check overlap and balance | [Diagnostics](diagnostics.md) |
| Put an interval on an estimate | [Uncertainty](uncertainty.md) |
| Probe unmeasured confounding | [Sensitivity](sensitivity.md) |
| Audit who a policy actually treats | [Fairness](fairness.md) |
| Write up a defensible result | [Reporting](reporting.md) |
| Plot diagnostics or results | [Plotting](plotting.md) |

## Result objects

Estimators return frozen dataclasses rather than bare floats, so an estimate
travels with the context needed to interpret it.

| Object | Carries |
|---|---|
| `EffectEstimate` | `estimate`, `estimator`, `estimand`, `n_observations` |
| `BootstrapResult` | `estimate`, `lower`, `upper`, `std_error`, `confidence_level` |
| `MatchingResult` | `effect`, `matched_data`, `dropped_units` |
| `IVResult` | `effect`, `first_stage_f_stat`, `first_stage_r2`, `instrument_is_weak` |
| `DifferenceInDifferencesResult` | `effect`, group means, `pre_trend_slope_difference`, `pre_trend_p_value` |
| `RDDResult` | `estimate`, `cutoff`, `bandwidth`, `standard_error` |
| `SyntheticControlResult` | `estimated_effect`, `pre_treatment_rmse`, `weights`, `effects` |
| `CausalReport` | The full write-up: question through recommendation |

`estimand` is a field for a reason. An `EffectEstimate` from
`propensity_score_matching` reports the ATT and one from `aipw_ate` reports the
ATE; they are not interchangeable, and the object says so.
