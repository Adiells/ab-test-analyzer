"""Exception hierarchy for the AB Test Analyzer.

All custom exceptions inherit from AnalyzerError, making it easy
to catch any application-level error with a single except clause.
"""


class AnalyzerError(Exception):
    """Base exception for all AB Test Analyzer errors."""


# ── Ingestion ────────────────────────────────────────────────────────

class IngestionError(AnalyzerError):
    """Raised when the data source cannot be read or loaded."""


class FileNotFoundError_(AnalyzerError):
    """Raised when the specified dataset file does not exist."""


class UnsupportedFileFormatError(IngestionError):
    """Raised when the file format is not supported."""


# ── Validation ───────────────────────────────────────────────────────

class ValidationError(AnalyzerError):
    """Raised when the dataset fails validation checks."""


class MissingColumnError(ValidationError):
    """Raised when a required column is absent from the dataset."""


class InvalidDataTypeError(ValidationError):
    """Raised when a column contains an unexpected data type."""


# ── Preprocessing ────────────────────────────────────────────────────

class PreprocessingError(AnalyzerError):
    """Raised when data transformation fails."""


# ── Normalization ────────────────────────────────────────────────────

class NormalizationError(AnalyzerError):
    """Raised when dataset normalization fails."""


# ── Metrics ──────────────────────────────────────────────────────────

class MetricsComputationError(AnalyzerError):
    """Raised when a metric cannot be computed."""


# ── Statistics ───────────────────────────────────────────────────────

class StatisticalAnalysisError(AnalyzerError):
    """Raised when a statistical test fails or produces invalid results."""


# ── Decision ─────────────────────────────────────────────────────────

class DecisionError(AnalyzerError):
    """Raised when the decision engine cannot produce a recommendation."""


# ── LLM ──────────────────────────────────────────────────────────────

class LLMError(AnalyzerError):
    """Raised when LLM interaction fails."""


# ── Reporting ────────────────────────────────────────────────────────

class ReportGenerationError(AnalyzerError):
    """Raised when report generation fails."""


# ── Integrations ─────────────────────────────────────────────────────

class IntegrationError(AnalyzerError):
    """Raised when an external integration fails."""


class GoogleSheetsError(IntegrationError):
    """Raised when the Google Sheets integration fails."""
