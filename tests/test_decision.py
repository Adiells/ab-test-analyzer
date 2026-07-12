"""Unit tests for the decision engine."""

import pytest

from src.core.decision.decision import DecisionService
from src.models import (
    Decision,
    MetricsResult,
    Recommendation,
    StatisticalResult,
)


@pytest.fixture
def service() -> DecisionService:
    return DecisionService()


def _make_metrics(g1_revenue: float, g2_revenue: float) -> MetricsResult:
    return MetricsResult(summary={
        "Grupo 1": {"total_net_revenue": g1_revenue, "total_buyers": 100},
        "Grupo 2": {"total_net_revenue": g2_revenue, "total_buyers": 100},
    })


def _make_stats(
    *,
    significant: bool,
    p_value: float = 0.01,
    d: float = 0.8,
    magnitude: str = "large",
) -> StatisticalResult:
    return StatisticalResult(tests={
        "n_groups": 2,
        "group_names": ["Grupo 1", "Grupo 2"],
        "target_metric": "net_revenue",
        "normality": {},
        "primary_test": {
            "method": "Welch's t-test",
            "statistic": 3.5,
            "p_value": p_value,
            "significant": significant,
            "cohens_d": {"d": d, "magnitude": magnitude},
            "confidence_interval": {
                "mean_difference": 100.0,
                "lower": 50.0,
                "upper": 150.0,
                "confidence_level": 0.95,
            },
        },
    })


class TestDecisionEngine:
    """Tests for DecisionService.decide."""

    def test_returns_decision(self, service: DecisionService) -> None:
        result = service.decide(
            _make_metrics(1000, 500),
            _make_stats(significant=True, d=0.8),
        )
        assert isinstance(result, Decision)

    def test_scale_when_significant_and_practical(
        self, service: DecisionService,
    ) -> None:
        result = service.decide(
            _make_metrics(1000, 500),
            _make_stats(significant=True, d=0.8, magnitude="large"),
        )
        assert result.recommendation == Recommendation.SCALE_TREATMENT

    def test_keep_control_when_significant_but_no_practical_effect(
        self, service: DecisionService,
    ) -> None:
        result = service.decide(
            _make_metrics(1000, 950),
            _make_stats(significant=True, d=0.05, magnitude="negligible"),
        )
        assert result.recommendation == Recommendation.KEEP_CONTROL

    def test_collect_more_data_when_not_significant_but_effect(
        self, service: DecisionService,
    ) -> None:
        result = service.decide(
            _make_metrics(1000, 500),
            _make_stats(significant=False, p_value=0.08, d=0.5, magnitude="medium"),
        )
        assert result.recommendation == Recommendation.COLLECT_MORE_DATA

    def test_inconclusive_when_nothing(
        self, service: DecisionService,
    ) -> None:
        result = service.decide(
            _make_metrics(1000, 950),
            _make_stats(significant=False, p_value=0.6, d=0.05, magnitude="negligible"),
        )
        assert result.recommendation == Recommendation.INCONCLUSIVE

    def test_winning_variant_set(self, service: DecisionService) -> None:
        result = service.decide(
            _make_metrics(1000, 500),
            _make_stats(significant=True),
        )
        assert result.winning_variant == "Grupo 1"

    def test_risks_populated(self, service: DecisionService) -> None:
        result = service.decide(
            _make_metrics(1000, 500),
            _make_stats(significant=False, p_value=0.08, d=0.5, magnitude="medium"),
        )
        assert len(result.risks) > 0

    def test_justification_not_empty(self, service: DecisionService) -> None:
        result = service.decide(
            _make_metrics(1000, 500),
            _make_stats(significant=True),
        )
        assert result.justification != ""
