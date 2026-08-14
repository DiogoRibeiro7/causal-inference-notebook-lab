# Security Policy

## Supported versions

This is a research and portfolio project, not production infrastructure. Only
the latest release receives fixes.

| Version | Supported |
|---|---|
| 0.1.x | Yes |
| < 0.1 | No |

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Use GitHub's [private vulnerability
reporting](https://github.com/DiogoRibeiro7/causal-inference-notebook-lab/security/advisories/new),
or email <dfr@esmad.ipp.pt> with "SECURITY" in the subject line.

Include what you found, how to reproduce it, and what an attacker could do with
it. You can expect an acknowledgement within seven days, and an assessment
within thirty.

## Scope

The realistic attack surface here is small but not empty:

- **`scripts/prepare_lalonde_job_training_dataset.py` fetches a CSV over the
  network** and parses it with pandas. A compromised or substituted upstream
  file is the most plausible supply-chain concern in this repository.
- **Notebooks execute arbitrary code.** Never run a notebook from an untrusted
  fork without reading it first — this is true of every Jupyter project, but
  worth stating.
- **Dependency vulnerabilities** in the scientific Python stack. Dependabot
  watches these; report anything it misses.

Out of scope: the statistical correctness of an estimator. That is a bug, and
an important one — but it belongs in a public issue, not a security advisory.
