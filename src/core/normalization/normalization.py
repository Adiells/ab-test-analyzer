"""Data normalization service.

Standardizes raw dataset column names before validation so the rest of
the pipeline can operate on a canonical schema.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src.exceptions import NormalizationError
from src.logging_config import get_logger

logger = get_logger(__name__)

CANONICAL_COLUMN_NAMES: dict[str, str] = {
    "data": "date",
    "grupos_de_usuarios": "user_group",
    "grupo_de_usuarios": "user_group",
    "grupos_de_usuario": "user_group",
    "grupo_de_usuario": "user_group",
    "parceiro": "partner",
    "compradores": "buyers",
    "comissao": "commission",
    "cashback": "cashback",
    "vendas_totais": "total_sales",
}


class NormalizationService:
    """Normalizes dataset column names to a canonical schema."""

    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *data* with standardized column names.

        Args:
            data: Raw experiment DataFrame.

        Returns:
            A copy of the DataFrame with normalized columns.

        Raises:
            NormalizationError: If normalization cannot be completed.
        """
        logger.info("Normalizing dataset with %d rows", len(data))

        if data.empty:
            raise NormalizationError("Dataset is empty")

        renamed_columns: dict[str, str] = {}
        seen_columns: set[str] = set()

        for original_name in data.columns:
            normalized_key = self._normalize_key(str(original_name))
            canonical_name = CANONICAL_COLUMN_NAMES.get(
                normalized_key,
                normalized_key,
            )

            if canonical_name in seen_columns:
                raise NormalizationError(
                    f"Duplicate column after normalization: {canonical_name}"
                )

            renamed_columns[original_name] = canonical_name
            seen_columns.add(canonical_name)

        normalized = data.copy()
        normalized = normalized.rename(columns=renamed_columns)

        if renamed_columns != {column: column for column in data.columns}:
            logger.info("Normalized columns: %s", renamed_columns)

        return normalized

    @staticmethod
    def _normalize_key(value: str) -> str:
        """Convert a raw column label into a lookup-friendly key."""
        normalized = unicodedata.normalize("NFKD", value)
        normalized = normalized.encode("ascii", "ignore").decode("ascii")
        normalized = normalized.strip().lower()
        normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
        return normalized.strip("_")
