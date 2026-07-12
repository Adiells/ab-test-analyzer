"""Shared data models used across pipeline stages.

Each dataclass represents a well-defined intermediate result passed
between pipeline components.  Using dataclasses rather than raw dicts
gives us type safety and self-documenting code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


# ── Enums ────────────────────────────────────────────────────────────

class Recommendation(Enum):
    """Possible outcomes of the decision engine."""

    SCALE_TREATMENT = "scale_treatment"
    KEEP_CONTROL = "keep_control"
    COLLECT_MORE_DATA = "collect_more_data"
    INCONCLUSIVE = "inconclusive"


# ── Pipeline data containers ─────────────────────────────────────────

@dataclass
class ExperimentDataset:
    """Raw dataset loaded during the ingestion stage."""

    data: pd.DataFrame
    source_path: str


@dataclass
class ValidationResult:
    """Output of the validation stage."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class MetricsResult:
    """Computed business metrics for both experiment groups."""

    summary: dict[str, object] = field(default_factory=dict)


@dataclass
class StatisticalResult:
    """Output of the statistical analysis stage."""

    tests: dict[str, object] = field(default_factory=dict)


@dataclass
class Decision:
    """Final decision produced by the decision engine."""

    recommendation: Recommendation = Recommendation.INCONCLUSIVE
    winning_variant: str = ""
    confidence: float = 0.0
    justification: str = ""
    risks: list[str] = field(default_factory=list)
    metrics_summary: MetricsResult = field(default_factory=MetricsResult)
    statistical_summary: StatisticalResult = field(default_factory=StatisticalResult)


@dataclass
class ExperimentReport:
    """Complete experiment report ready for output."""

    experiment_name: str = ""
    description: str = ""
    partner: str = ""
    period: str = ""
    n_variants: int = 0
    statistical_significance: bool = False
    decision: Decision = field(default_factory=Decision)
    narrative: str = ""
    timestamp: str = ""
