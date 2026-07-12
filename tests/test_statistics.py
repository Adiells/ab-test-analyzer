"""Unit tests for the statistical analysis service."""

import numpy as np
import pandas as pd
import pytest

from src.config import StatisticsConfig
from src.core.statistics.statistics import StatisticsService
from src.models import MetricsResult, StatisticalResult


@pytest.fixture
def service() -> StatisticsService:
    return StatisticsService(StatisticsConfig())


def _make_two_group_df(seed: int = 42) -> pd.DataFrame:
    """Build a two-group preprocessed DataFrame."""
    rng = np.random.default_rng(seed)
    n = 45
    return pd.DataFrame({
        "user_group": ["Grupo 1"] * n + ["Grupo 2"] * n,
        "net_revenue": np.concatenate([
            rng.normal(800, 100, n),
            rng.normal(600, 100, n),
        ]),
    })


def _make_three_group_df(seed: int = 42) -> pd.DataFrame:
    """Build a three-group preprocessed DataFrame."""
    rng = np.random.default_rng(seed)
    n = 30
    return pd.DataFrame({
        "user_group": ["Grupo 1"] * n + ["Grupo 2"] * n + ["Grupo 3"] * n,
        "net_revenue": np.concatenate([
            rng.normal(800, 100, n),
            rng.normal(600, 100, n),
            rng.normal(700, 100, n),
        ]),
    })


class TestStatisticsTwoGroups:
    """Tests for two-group statistical analysis."""

    def test_returns_statistical_result(self, service: StatisticsService) -> None:
        result = service.analyze(_make_two_group_df(), MetricsResult())
        assert isinstance(result, StatisticalResult)

    def test_detects_two_groups(self, service: StatisticsService) -> None:
        result = service.analyze(_make_two_group_df(), MetricsResult())
        assert result.tests["n_groups"] == 2

    def test_primary_test_present(self, service: StatisticsService) -> None:
        result = service.analyze(_make_two_group_df(), MetricsResult())
        assert "primary_test" in result.tests

    def test_significance_detected(self, service: StatisticsService) -> None:
        result = service.analyze(_make_two_group_df(), MetricsResult())
        test = result.tests["primary_test"]
        # Groups have clearly different means → should be significant
        assert test["significant"] is True

    def test_cohens_d_present(self, service: StatisticsService) -> None:
        result = service.analyze(_make_two_group_df(), MetricsResult())
        test = result.tests["primary_test"]
        assert "cohens_d" in test
        assert "d" in test["cohens_d"]

    def test_confidence_interval_present(self, service: StatisticsService) -> None:
        result = service.analyze(_make_two_group_df(), MetricsResult())
        test = result.tests["primary_test"]
        assert "confidence_interval" in test
        ci = test["confidence_interval"]
        assert "lower" in ci
        assert "upper" in ci


class TestStatisticsThreeGroups:
    """Tests for three-group statistical analysis."""

    def test_detects_three_groups(self, service: StatisticsService) -> None:
        result = service.analyze(_make_three_group_df(), MetricsResult())
        assert result.tests["n_groups"] == 3

    def test_omnibus_test_present(self, service: StatisticsService) -> None:
        result = service.analyze(_make_three_group_df(), MetricsResult())
        assert "omnibus_test" in result.tests

    def test_pairwise_when_significant(self, service: StatisticsService) -> None:
        result = service.analyze(_make_three_group_df(), MetricsResult())
        omnibus = result.tests["omnibus_test"]
        if omnibus["significant"]:
            assert "pairwise" in result.tests
            assert len(result.tests["pairwise"]) == 3  # C(3,2) = 3 pairs

    def test_normality_tested(self, service: StatisticsService) -> None:
        result = service.analyze(_make_three_group_df(), MetricsResult())
        assert "normality" in result.tests
        assert len(result.tests["normality"]) == 3
