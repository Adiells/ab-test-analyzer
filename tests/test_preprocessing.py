"""Unit tests for the preprocessing service."""

import pandas as pd
import pytest

from src.core.preprocessing.preprocessing import PreprocessingService


@pytest.fixture
def service() -> PreprocessingService:
    return PreprocessingService()


def _make_raw_df() -> pd.DataFrame:
    """Build a raw DataFrame mimicking normalization output."""
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "user_group": ["Grupo 1", "Grupo 2"],
        "partner": ["A", "A"],
        "buyers": [100, 200],
        "commission": ["R$ 1.000", "R$ 2.000"],
        "cashback": ["R$ 500", "R$ 800"],
        "total_sales": ["R$ 10.000", "R$ 20.000"],
    })


class TestPreprocessing:
    """Tests for PreprocessingService.preprocess."""

    def test_returns_dataframe(self, service: PreprocessingService) -> None:
        result = service.preprocess(_make_raw_df())
        assert isinstance(result, pd.DataFrame)

    def test_date_converted(self, service: PreprocessingService) -> None:
        result = service.preprocess(_make_raw_df())
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_monetary_converted(self, service: PreprocessingService) -> None:
        result = service.preprocess(_make_raw_df())
        assert result["commission"].dtype == float
        assert result["cashback"].dtype == float
        assert result["total_sales"].dtype == float

    def test_monetary_values_correct(self, service: PreprocessingService) -> None:
        result = service.preprocess(_make_raw_df())
        assert result["commission"].iloc[0] == 1000.0
        assert result["total_sales"].iloc[1] == 20000.0

    def test_derived_columns_present(self, service: PreprocessingService) -> None:
        result = service.preprocess(_make_raw_df())
        expected = [
            "net_revenue", "sales_per_buyer", "commission_per_buyer",
            "cashback_per_buyer", "net_revenue_per_buyer",
            "commission_pct", "cashback_pct", "net_margin",
        ]
        for col in expected:
            assert col in result.columns, f"Missing derived column: {col}"

    def test_net_revenue_formula(self, service: PreprocessingService) -> None:
        result = service.preprocess(_make_raw_df())
        # net_revenue = commission - cashback = 1000 - 500 = 500
        assert result["net_revenue"].iloc[0] == 500.0

    def test_does_not_modify_input(self, service: PreprocessingService) -> None:
        original = _make_raw_df()
        original_cols = set(original.columns)
        service.preprocess(original)
        assert set(original.columns) == original_cols
