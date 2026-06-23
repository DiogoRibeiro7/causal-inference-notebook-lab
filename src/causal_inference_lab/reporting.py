"""Reusable causal analysis report object."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CausalReport:
    """Structured causal analysis report with explicit assumptions and uncertainty."""

    question: str
    estimand: str
    identification_assumptions: tuple[str, ...]
    estimator: str
    diagnostics: dict[str, object]
    uncertainty: dict[str, object]
    results: dict[str, object]
    limitations: tuple[str, ...]
    recommendation: str

    def to_markdown(self) -> str:
        """Render the report as Markdown with no hidden assumptions."""

        diagnostics = "\n".join(
            f"- {name}: {value}" for name, value in sorted(self.diagnostics.items())
        )
        uncertainty = "\n".join(
            f"- {name}: {value}" for name, value in sorted(self.uncertainty.items())
        )
        results = "\n".join(f"- {name}: {value}" for name, value in sorted(self.results.items()))
        assumptions = "\n".join(f"- {item}" for item in self.identification_assumptions)
        limitations = "\n".join(f"- {item}" for item in self.limitations)

        return (
            "# Causal Analysis Report\n\n"
            f"## Question\n{self.question}\n\n"
            f"## Estimand\n{self.estimand}\n\n"
            "## Identification assumptions\n"
            f"{assumptions}\n\n"
            f"## Estimator\n{self.estimator}\n\n"
            "## Diagnostics\n"
            f"{diagnostics}\n\n"
            "## Uncertainty\n"
            f"{uncertainty}\n\n"
            "## Results\n"
            f"{results}\n\n"
            "## Limitations\n"
            f"{limitations}\n\n"
            f"## Recommendation\n{self.recommendation}"
        )
