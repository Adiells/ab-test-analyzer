"""Data validation service.

Responsible for checking dataset integrity: required columns,
data types, missing values, and business-specific constraints.
"""

from __future__ import annotations

import re

import pandas as pd

from src.logging_config import get_logger
from src.models import ValidationResult

logger = get_logger(__name__)

REQUIRED_COLUMNS = (
    "date",
    "user_group",
    "partner",
    "buyers",
    "commission",
    "cashback",
    "total_sales",
)

_MONETARY_PATTERN = re.compile(r"^R\$\s*[\d.]+$")


class ValidationService:
    """Validates an experiment dataset before analysis."""

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Run all validation checks on *data*.

        Args:
            data: The normalized experiment DataFrame.

        Returns:
            A ``ValidationResult`` summarising any errors or warnings.
        """
        logger.info("Validation started — %d rows", len(data))

        errors: list[str] = []
        warnings: list[str] = []

        self._check_empty(data, errors)
        if errors:
            return ValidationResult(is_valid=False, errors=errors)

        self._check_required_columns(data, errors)
        if errors:
            return ValidationResult(is_valid=False, errors=errors)

        self._check_null_values(data, errors)
        self._check_minimum_groups(data, errors)
        self._check_single_partner(data, errors)
        self._check_duplicate_rows(data, errors, warnings)
        self._check_dates(data, errors)
        self._check_numeric_values(data, errors)
        self._check_negative_buyers(data, errors)

        is_valid = len(errors) == 0
        logger.info(
            "Validation completed — valid=%s, errors=%d, warnings=%d",
            is_valid, len(errors), len(warnings),
        )
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)

    # ── Individual checks ────────────────────────────────────────────

    @staticmethod
    def _check_empty(data: pd.DataFrame, errors: list[str]) -> None:
        """Reject empty datasets."""
        if data.empty:
            errors.append("Dataset is empty.")

    @staticmethod
    def _check_required_columns(data: pd.DataFrame, errors: list[str]) -> None:
        """Ensure every required column is present."""
        missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            errors.append(f"Missing required columns: {', '.join(missing)}")

    @staticmethod
    def _check_null_values(data: pd.DataFrame, errors: list[str]) -> None:
        """Flag required columns that contain missing values."""
        null_cols = [
            c for c in REQUIRED_COLUMNS
            if c in data.columns and data[c].isna().any()
        ]
        if null_cols:
            errors.append(
                f"Required columns contain missing values: {', '.join(null_cols)}"
            )

    @staticmethod
    def _check_minimum_groups(data: pd.DataFrame, errors: list[str]) -> None:
        """At least two experiment groups are required."""
        if "user_group" in data.columns:
            n_groups = data["user_group"].nunique()
            if n_groups < 2:
                errors.append(
                    f"At least 2 experiment groups required, found {n_groups}."
                )

    @staticmethod
    def _check_single_partner(data: pd.DataFrame, errors: list[str]) -> None:
        """Each dataset must contain exactly one partner."""
        if "partner" in data.columns:
            n_partners = data["partner"].nunique()
            if n_partners != 1:
                errors.append(
                    f"Expected exactly 1 partner, found {n_partners}: "
                    f"{data['partner'].unique().tolist()}"
                )

    @staticmethod
    def _check_duplicate_rows(
        data: pd.DataFrame,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Detect fully duplicated rows."""
        n_duplicates = data.duplicated().sum()
        if n_duplicates > 0:
            warnings.append(f"Found {n_duplicates} duplicate row(s).")

    @staticmethod
    def _check_dates(data: pd.DataFrame, errors: list[str]) -> None:
        """Verify that the date column can be parsed."""
        if "date" not in data.columns:
            return
        try:
            pd.to_datetime(data["date"], format="mixed")
        except (ValueError, TypeError):
            errors.append("Column 'date' contains unparseable date values.")

    @staticmethod
    def _check_numeric_values(data: pd.DataFrame, errors: list[str]) -> None:
        """Monetary columns must match the expected Brazilian format."""
        monetary_cols = ["commission", "cashback", "total_sales"]
        for col in monetary_cols:
            if col not in data.columns:
                continue
            invalid = data[col].dropna().apply(
                lambda v: isinstance(v, str) and not _MONETARY_PATTERN.match(v)
            )
            if invalid.any():
                n_invalid = invalid.sum()
                errors.append(
                    f"Column '{col}' has {n_invalid} value(s) that don't "
                    f"match the expected monetary format (e.g. 'R$ 1.234')."
                )

    @staticmethod
    def _check_negative_buyers(data: pd.DataFrame, errors: list[str]) -> None:
        """Buyer counts must not be negative."""
        if "buyers" not in data.columns:
            return
        try:
            buyers = pd.to_numeric(data["buyers"], errors="coerce")
            if (buyers < 0).any():
                errors.append("Column 'buyers' contains negative values.")
        except (ValueError, TypeError):
            errors.append("Column 'buyers' contains non-numeric values.")
