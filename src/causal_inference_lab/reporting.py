"""Reusable causal analysis report object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    def __post_init__(self) -> None:
        """Validate fields to ensure report sections are explicit and parseable."""

        _validate_non_empty_string(self.question, "question")
        _validate_non_empty_string(self.estimand, "estimand")
        _validate_non_empty_string(self.estimator, "estimator")
        _validate_non_empty_string(self.recommendation, "recommendation")
        _validate_assumption_tuple(self.identification_assumptions, "identification_assumptions")
        _validate_assumption_tuple(self.limitations, "limitations")
        _validate_report_map(self.diagnostics, "diagnostics")
        _validate_report_map(self.uncertainty, "uncertainty")
        _validate_report_map(self.results, "results")

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
            f"{diagnostics or '- (not reported)'}\n\n"
            "## Uncertainty\n"
            f"{uncertainty or '- (not reported)'}\n\n"
            "## Results\n"
            f"{results or '- (not reported)'}\n\n"
            "## Limitations\n"
            f"{limitations}\n\n"
            f"## Recommendation\n{self.recommendation}"
        )


def _validate_non_empty_string(value: str, name: str) -> None:
    """Validate a required non-empty string field."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")


def _validate_assumption_tuple(values: tuple[str, ...], name: str) -> None:
    """Validate assumptions/limitations as non-empty string tuples."""

    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple.")
    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one item.")
    if not all(isinstance(item, str) and item.strip() for item in values):
        raise ValueError(f"{name} must be a tuple of non-empty strings.")


def _validate_report_map(mapping: dict[str, Any], name: str) -> None:
    """Validate report mapping sections."""

    if not isinstance(mapping, dict):
        raise TypeError(f"{name} must be a dictionary.")
    if not all(isinstance(key, str) and key.strip() for key in mapping):
        raise TypeError(f"{name} must have non-empty string keys.")
