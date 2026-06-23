from __future__ import annotations

from pathlib import Path

from causal_inference_lab.reporting import CausalReport


def test_causal_report_renders_markdown_with_assumptions() -> None:
    report = CausalReport(
        question="What is the effect of treatment on earnings?",
        estimand="ATE",
        identification_assumptions=(
            "Conditional ignorability given covariates",
            "Common support across treatment groups",
            "SUTVA",
        ),
        estimator="AIPW",
        diagnostics={"abs_smd": 0.07},
        uncertainty={"bootstrap_ci_low": -0.5, "bootstrap_ci_high": 1.2},
        results={"estimate": 0.38},
        limitations=("Networked causal channels not observed", "Generalization uncertainty"),
        recommendation="Use with caution and run placebo checks.",
    )

    markdown = report.to_markdown()

    assert "# Causal Analysis Report" in markdown
    assert "## Identification assumptions" in markdown
    assert "Conditional ignorability" in markdown
    assert "AIPW" in markdown
    assert "## Recommendation" in markdown
