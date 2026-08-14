# Assumptions matrix

Every estimator here identifies a causal effect only under conditions that the
data cannot verify. This page states them, names what each estimator actually
targets, and describes how each one fails.

Contributors adding an estimator are expected to add a row.

## Shared assumptions

These apply to essentially everything below.

| Assumption | Statement | If violated |
|---|---|---|
| **SUTVA** | One unit's treatment does not affect another's outcome, and there is one version of the treatment | Interference biases every estimator here; none of them detect it |
| **Consistency** | The observed outcome equals the potential outcome under the treatment received | The estimand stops being well defined |
| **Positivity / overlap** | Every unit has a non-zero probability of either treatment | Weights explode, models extrapolate; check before estimating |
| **No measurement error** | Treatment, outcome, and covariates are recorded correctly | Attenuates effects; not modelled anywhere in this project |

## Selection on observables

Shared additional assumption: **conditional ignorability** — treatment is
independent of potential outcomes given covariates. This is untestable. Its
plausibility is a subject-matter argument, not a statistical one.

| Estimator | Estimand | Additional requirements | Failure mode |
|---|---|---|---|
| `difference_in_means` | ATE | Randomised assignment | Under confounding, returns the confounded contrast — useful only as a baseline |
| `g_computation_ate` | ATE | Correctly specified outcome model | Misspecification biases the estimate with no warning sign |
| `ipw_ate` | ATE | Correctly specified propensity model; adequate overlap | Extreme weights inflate variance; `clip` trades bias for stability |
| `aipw_ate` | ATE | *Either* the outcome or the propensity model correct | Both wrong, or overlap poor, and double robustness gives nothing |
| `double_machine_learning_ate` | ATE | Cross-fitting; nuisance models converge fast enough | Overfitted nuisances reintroduce the regularization bias cross-fitting removes |
| `propensity_score_matching` | ATT | Overlap in propensity; caliper not too tight | Unmatched treated units are dropped — the estimand quietly changes |
| `nearest_neighbour_matching` | ATT | Meaningful distance metric across covariates | Poor matches in high dimensions; check the balance table, not the estimate |

!!! danger "Matching changes the estimand"
    `MatchingResult.dropped_units` is not diagnostic trivia. Dropping treated
    units means you are no longer estimating the ATT for the original
    population, but for whichever units happened to find a match. Report it.

## Heterogeneous effects

| Estimator | Estimand | Additional requirements | Failure mode |
|---|---|---|---|
| `SMetaLearner` | CATE | Conditional ignorability; model can represent the interaction | Regularization shrinks the treatment indicator, biasing effects toward zero |
| `TMetaLearner` | CATE | Enough data in *both* arms | High variance when one arm is small |
| `XMetaLearner` | CATE | As above, plus a reasonable propensity model | More moving parts, more ways to be wrong; still needs overlap |

CATE estimates carry substantially more variance than ATE estimates. Validating
against `true_ite` on synthetic data is the cheapest guard against reading noise
as personalization.

## Designs that relax ignorability

| Estimator | Estimand | Key assumptions | Failure mode |
|---|---|---|---|
| `instrumental_variables_ate` | **LATE** (compliers only) | Relevance; exclusion restriction; independence; monotonicity | Weak instrument gives severe bias and misleading intervals — check `first_stage_f_stat` and `instrument_is_weak`; the exclusion restriction is untestable |
| `local_linear_rdd` | Effect **at the cutoff** | Continuity of potential outcomes at the cutoff; no manipulation of the running variable | Bandwidth choice drives the estimate — run `rdd_bandwidth_sensitivity`; sorting around the cutoff invalidates the design |
| `difference_in_differences` | ATT | Parallel trends; no anticipation; stable composition | Pre-trend divergence invalidates it — check `pre_trend_p_value`, and remember a passing test is weak evidence, not proof |
| `fit_synthetic_control` | Effect on the one treated unit | Pre-treatment fit; donors untreated and not spillover-affected | Good pre-treatment fit does not guarantee a valid counterfactual; inference with one treated unit is inherently limited |

!!! warning "IV does not estimate the ATE"
    `instrumental_variables_ate` returns the local average treatment effect —
    the effect among units whose treatment status responds to the instrument. If
    compliers differ systematically from the population, the LATE does not
    generalise, and presenting it as an average effect is wrong.

## Diagnostics and what they cannot do

| Tool | Checks | Does not check |
|---|---|---|
| `diagnostics` — overlap, standardized mean differences | Balance on **measured** covariates | Balance on unmeasured confounders |
| `sensitivity` — omitted confounder simulation | How strong a confounder would need to be to overturn the result | Whether such a confounder exists |
| `bootstrap_ate` | Sampling variability | Bias from misspecification or violated identification |
| `rdd_bandwidth_sensitivity` | Stability across bandwidths | Whether the running variable was manipulated |

The pattern is consistent: diagnostics can falsify an analysis, never validate
one. Balance on observables is evidence that adjustment worked, not evidence
that ignorability holds.

## Reporting

`CausalReport` requires question, estimand, identification assumptions,
estimator, diagnostics, uncertainty, results, limitations, and recommendation.
Nothing enforces that the fields are *honest* — but a report with an empty
limitations tuple is a red flag in review.
