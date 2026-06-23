from __future__ import annotations

import pytest

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


def test_causal_report_rejects_empty_strings_and_invalid_sections() -> None:
    with pytest.raises(ValueError, match="question must not be empty."):
        CausalReport(
            question=" ",
            estimand="ATE",
            identification_assumptions=("A1",),
            estimator="AIPW",
            diagnostics={"abs_smd": 0.07},
            uncertainty={"se": 0.1},
            results={"estimate": 0.2},
            limitations=("L1",),
            recommendation="Proceed. ",
        )

    with pytest.raises(TypeError, match="identification_assumptions must be a tuple."):
        CausalReport(
            question="Q",
            estimand="ATE",
            identification_assumptions=["A1"],  # type: ignore[arg-type]
            estimator="AIPW",
            diagnostics={"abs_smd": 0.07},
            uncertainty={"se": 0.1},
            results={"estimate": 0.2},
            limitations=("L1",),
            recommendation="Proceed.",
        )

    with pytest.raises(TypeError, match="diagnostics must be a dictionary."):
        CausalReport(
            question="Q",
            estimand="ATE",
            identification_assumptions=("A1",),
            estimator="AIPW",
            diagnostics=[],  # type: ignore[arg-type]
            uncertainty={"se": 0.1},
            results={"estimate": 0.2},
            limitations=("L1",),
            recommendation="Proceed.",
        )

    with pytest.raises(ValueError, match="limitations must contain at least one item."):
        CausalReport(
            question="Q",
            estimand="ATE",
            identification_assumptions=("A1",),
            estimator="AIPW",
            diagnostics={"abs_smd": 0.07},
            uncertainty={"se": 0.1},
            results={"estimate": 0.2},
            limitations=(),
            recommendation="Proceed.",
        )


def test_causal_report_renders_placeholders_for_empty_sections() -> None:
    report = CausalReport(
        question="Q",
        estimand="ATE",
        identification_assumptions=("A1",),
        estimator="AIPW",
        diagnostics={},
        uncertainty={},
        results={},
        limitations=("L1",),
        recommendation="Proceed cautiously.",
    )

    markdown = report.to_markdown()

    assert "(not reported)" in markdown
    assert "A1" in markdown
    assert "L1" in markdown
