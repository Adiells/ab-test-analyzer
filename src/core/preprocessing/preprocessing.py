"""Data preprocessing service.

Transforms raw validated data into a format suitable for metric
computation and statistical analysis.
"""

from __future__ import annotations

import pandas as pd

from src.exceptions import PreprocessingError
from src.logging_config import get_logger

logger = get_logger(__name__)


class PreprocessingService:
    """Applies data transformations to prepare for analysis."""

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform *data* into an analysis-ready DataFrame.

        Args:
            data: Validated experiment DataFrame.

        Returns:
            A cleaned and transformed DataFrame with derived columns.

        Raises:
            PreprocessingError: If any transformation step fails.
        """
        logger.info("Preprocessing started — %d rows", len(data))

        try:
            df = data.copy()
            df = self._convert_dates(df)
            df = self._convert_monetary_columns(df)
            df = self._compute_derived_columns(df)
        except PreprocessingError:
            raise
        except Exception as exc:
            logger.exception("Preprocessing failed")
            raise PreprocessingError(
                "Unexpected error during preprocessing."
            ) from exc

        logger.info("Preprocessing completed — %d columns", len(df.columns))
        return df

    # ── Internal steps ───────────────────────────────────────────────

    @staticmethod
    def _convert_dates(df: pd.DataFrame) -> pd.DataFrame:
        """Parse the date column to datetime."""
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _convert_monetary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Brazilian monetary strings to float."""
        monetary_columns = ["commission", "cashback", "total_sales"]
        for col in monetary_columns:
            df[col] = df[col].apply(self._parse_monetary_value).astype(float)
        return df

    @staticmethod
    def _parse_monetary_value(value: object) -> str:
        """Strip currency prefix and convert Brazilian number format.

        'R$ 10.273' → '10273'
        """
        if not isinstance(value, str):
            raise PreprocessingError(
                f"Expected monetary string, got {type(value).__name__}: {value}"
            )
        # Remove 'R$ ' prefix, strip dots (thousands separator)
        cleaned = value.replace(".", "").replace(",", ".")
        # Remove the currency prefix (e.g. 'R$ ')
        for prefix in ("R$ ", "R$"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
        return cleaned

    @staticmethod
    def _compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Add all derived business columns."""
        df["net_revenue"] = df["commission"] - df["cashback"]

        df["sales_per_buyer"] = (
            df["total_sales"] / df["buyers"]
        ).round(2)

        df["commission_per_buyer"] = (
            df["commission"] / df["buyers"]
        ).round(2)

        df["cashback_per_buyer"] = (
            df["cashback"] / df["buyers"]
        ).round(2)

        df["net_revenue_per_buyer"] = (
            df["net_revenue"] / df["buyers"]
        ).round(2)

        df["commission_pct"] = (
            df["commission"] / df["total_sales"]
        ).round(4)

        df["cashback_pct"] = (
            df["cashback"] / df["commission"]
        ).round(4)

        df["net_margin"] = (
            df["net_revenue"] / df["total_sales"]
        ).round(4)

        return df