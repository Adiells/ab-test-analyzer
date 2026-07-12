"""Data ingestion service.

Responsible for reading experiment datasets from the filesystem and
returning them in a structured format for downstream processing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.constants import SUPPORTED_FILE_EXTENSIONS
from src.exceptions import FileNotFoundError_, UnsupportedFileFormatError
from src.logging_config import get_logger
from src.models import ExperimentDataset

logger = get_logger(__name__)


class IngestionService:
    """Loads experiment data from supported file formats."""

    def load(self, file_path: str | Path) -> ExperimentDataset:
        """Read a dataset file and return an ``ExperimentDataset``.

        Args:
            file_path: Path to the dataset file.

        Returns:
            An ``ExperimentDataset`` wrapping the loaded DataFrame.

        Raises:
            FileNotFoundError_: If *file_path* does not exist.
            UnsupportedFileFormatError: If the file extension is not supported.
            NotImplementedError: Parsing logic not yet implemented.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError_(f"Dataset not found: {path}")

        if path.suffix.lower() not in SUPPORTED_FILE_EXTENSIONS:
            raise UnsupportedFileFormatError(
                f"Unsupported format '{path.suffix}'. "
                f"Supported: {', '.join(SUPPORTED_FILE_EXTENSIONS)}"
            )

        logger.info("Loading dataset from %s", path)

        return ExperimentDataset(data=pd.read_csv(path), source_path=str(path))