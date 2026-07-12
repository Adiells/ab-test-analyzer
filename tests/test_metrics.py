"""Unit tests for the metrics computation service."""

import pandas as pd
import pytest

from src.core.metrics.metrics import MetricsService
from src.models import MetricsResult


@pytest.fixture
def service() -> MetricsService:
    return MetricsService()


def _make_processed_df() -> pd.DataFrame:
    """Build a preprocessed DataFrame with numeric values."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
        "user_group": ["Grupo 1", "Grupo 1", "Grupo 2", "Grupo 2"],
        "partner": ["A", "A", "A", "A"],
        "buyers": [100, 200, 150, 250],
        "commission": [1000.0, 2000.0, 1500.0, 2500.0],
        "cashback": [500.0, 800.0, 700.0, 1200.0],
        "total_sales": [10000.0, 20000.0, 15000.0, 25000.0],
        "net_revenue": [500.0, 1200.0, 800.0, 1300.0],
    })


class TestMetrics:
    """Tests for MetricsService.compute."""

    def test_returns_metrics_result(self, service: MetricsService) -> None:
        result = service.compute(_make_processed_df())
        assert isinstance(result, MetricsResult)

    def test_groups_detected(self, service: MetricsService) -> None:
        result = service.compute(_make_processed_df())
        assert "Grupo 1" in result.summary
        assert "Grupo 2" in result.summary

    def test_total_buyers(self, service: MetricsService) -> None:
        result = service.compute(_make_processed_df())
        assert result.summary["Grupo 1"]["total_buyers"] == 300
        assert result.summary["Grupo 2"]["total_buyers"] == 400

    def test_total_net_revenue(self, service: MetricsService) -> None:
        result = service.compute(_make_processed_df())
        assert result.summary["Grupo 1"]["total_net_revenue"] == 1700.0
        assert result.summary["Grupo 2"]["total_net_revenue"] == 2100.0

    def test_per_buyer_metrics(self, service: MetricsService) -> None:
        result = service.compute(_make_processed_df())
        g1 = result.summary["Grupo 1"]
        assert "sales_per_buyer" in g1
        assert "net_revenue_per_buyer" in g1

    def test_financial_ratios(self, service: MetricsService) -> None:
        result = service.compute(_make_processed_df())
        g1 = result.summary["Grupo 1"]
        assert "commission_rate" in g1
        assert "cashback_rate" in g1
        assert "net_margin" in g1

    def test_experiment_days(self, service: MetricsService) -> None:
        result = service.compute(_make_processed_df())
        assert result.summary["Grupo 1"]["experiment_days"] == 2
