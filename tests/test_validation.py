"""Unit tests for the validation service."""

import pandas as pd
import pytest

from src.core.validation.validation import ValidationService
from src.models import ValidationResult


@pytest.fixture
def service() -> ValidationService:
    return ValidationService()


def _make_df(**overrides: object) -> pd.DataFrame:
    """Build a minimal valid DataFrame, applying *overrides*."""
    base = {
        "date": ["2024-01-01", "2024-01-02"],
        "user_group": ["Grupo 1", "Grupo 2"],
        "partner": ["Partner A", "Partner A"],
        "buyers": [100, 200],
        "commission": ["R$ 1.000", "R$ 2.000"],
        "cashback": ["R$ 500", "R$ 1.000"],
        "total_sales": ["R$ 10.000", "R$ 20.000"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestValidation:
    """Tests for ValidationService.validate."""

    def test_valid_dataset(self, service: ValidationService) -> None:
        result = service.validate(_make_df())
        assert result.is_valid
        assert result.errors == []

    def test_empty_dataset(self, service: ValidationService) -> None:
        result = service.validate(pd.DataFrame())
        assert not result.is_valid
        assert any("empty" in e.lower() for e in result.errors)

    def test_missing_columns(self, service: ValidationService) -> None:
        df = _make_df()
        df = df.drop(columns=["buyers", "cashback"])
        result = service.validate(df)
        assert not result.is_valid
        assert any("buyers" in e for e in result.errors)

    def test_less_than_two_groups(self, service: ValidationService) -> None:
        result = service.validate(_make_df(user_group=["Grupo 1", "Grupo 1"]))
        assert not result.is_valid
        assert any("2 experiment groups" in e for e in result.errors)

    def test_multiple_partners(self, service: ValidationService) -> None:
        result = service.validate(
            _make_df(partner=["Partner A", "Partner B"])
        )
        assert not result.is_valid
        assert any("1 partner" in e for e in result.errors)

    def test_null_values(self, service: ValidationService) -> None:
        result = service.validate(
            _make_df(buyers=[100, None])
        )
        assert not result.is_valid
        assert any("missing values" in e for e in result.errors)

    def test_negative_buyers(self, service: ValidationService) -> None:
        result = service.validate(_make_df(buyers=[-1, 200]))
        assert not result.is_valid
        assert any("negative" in e.lower() for e in result.errors)

    def test_duplicate_rows_warning(self, service: ValidationService) -> None:
        df = _make_df()
        # Append a true duplicate of the first row
        dup_row = df.iloc[[0]].copy()
        df = pd.concat([df, dup_row], ignore_index=True)
        result = service.validate(df)
        assert any("duplicate" in w.lower() for w in result.warnings)
