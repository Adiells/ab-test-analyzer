"""Google Sheets client service.

Registers experiment results in a tracking spreadsheet so that the
team has a consolidated view of all experiments and their outcomes.

When Google Sheets credentials are unavailable, automatically falls
back to appending to a local CSV file with the same schema.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.config import GoogleSheetsConfig
from src.constants import OUTPUT_DIR
from src.exceptions import GoogleSheetsError
from src.logging_config import get_logger
from src.models import ExperimentReport, Recommendation

logger = get_logger(__name__)

_CSV_FILENAME = "experiment_log.csv"

_RECOMMENDATION_LABELS: dict[Recommendation, str] = {
    Recommendation.SCALE_TREATMENT: "Scale winner",
    Recommendation.KEEP_CONTROL: "Keep current experiment",
    Recommendation.COLLECT_MORE_DATA: "Collect more data",
    Recommendation.INCONCLUSIVE: "Inconclusive",
}

_CSV_HEADERS = [
    "experiment_name",
    "report_file_path",
    "partner",
    "period",
    "n_variants",
    "winning_variant",
    "statistical_significance",
    "decision",
    "description",
    "timestamp",
]


class GoogleSheetsService:
    """Writes experiment results to Google Sheets."""

    def __init__(self, config: GoogleSheetsConfig) -> None:
        self._config = config

    def register_experiment(self, report: ExperimentReport) -> None:
        """Append an experiment result row to the tracking spreadsheet.

        If credentials are configured and valid, writes to Google Sheets.
        Otherwise, writes to a local CSV file under the output directory.

        Args:
            report: The completed experiment report.

        Raises:
            GoogleSheetsError: If Google Sheets write fails unexpectedly.
        """
        logger.info(
            "Registering experiment '%s'", report.experiment_name,
        )

        row = self._build_row(report)

        if self._credentials_available():
            try:
                if self._is_spreadsheet_empty():
                    self._write_to_sheets(_CSV_HEADERS)
                self._write_to_sheets(row)
            except Exception as exc:
                logger.warning(
                    "Google Sheets integration failed — falling back to CSV. Error: %s", exc
                )
                self._write_to_csv(row)
        else:
            logger.info(
                "Google Sheets credentials not available — writing to CSV"
            )
            self._write_to_csv(row)

    # ── Row construction ─────────────────────────────────────────────

    @staticmethod
    def _build_row(report: ExperimentReport) -> dict[str, str]:
        """Build a flat row dict from the report."""
        d = report.decision
        safe_name = f"{report.experiment_name.replace(' ', '_').lower()}_report.md"

        return {
            "experiment_name": report.experiment_name,
            "report_file_path": safe_name,
            "description": report.narrative,
            "partner": report.partner,
            "period": report.period,
            "n_variants": str(report.n_variants),
            "winning_variant": d.winning_variant or "N/A",
            "statistical_significance": (
                "Yes" if report.statistical_significance else "No"
            ),
            "decision": _RECOMMENDATION_LABELS.get(
                d.recommendation, d.recommendation.value,
            ),
            "timestamp": report.timestamp,
        }

    # ── Credential check ─────────────────────────────────────────────

    def _credentials_available(self) -> bool:
        """Check if Google Sheets credentials file exists."""
        creds_path = Path(self._config.credentials_path)
        return (
            bool(self._config.spreadsheet_id)
            and creds_path.exists()
        )

    # ── Google Sheets write ──────────────────────────────────────────

    def _is_spreadsheet_empty(self) -> bool:
        """Check if the Google Sheets spreadsheet is empty."""
        try:
            import gspread
            from google.oauth2.service_account import Credentials

            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]
            creds = Credentials.from_service_account_file(
                str(self._config.credentials_path),
                scopes=scopes,
            )
            client = gspread.authorize(creds)
            sheet = client.open_by_key(self._config.spreadsheet_id).sheet1

            value = sheet.acell("A1").value

            return not value or value.strip() == ""

        except Exception as exc:
            logger.error("Google Sheets write failed: %s", exc)
            raise GoogleSheetsError(
                f"Failed to write to Google Sheets: {exc}"
            ) from exc

    def _write_to_sheets(self, row: dict[str, str] | list[str]) -> None:
        """Append *row* to the configured Google Sheets spreadsheet."""
        try:
            import gspread
            from google.oauth2.service_account import Credentials

            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]
            creds = Credentials.from_service_account_file(
                str(self._config.credentials_path),
                scopes=scopes,
            )
            client = gspread.authorize(creds)
            sheet = client.open_by_key(self._config.spreadsheet_id).sheet1

            if isinstance(row, dict):
                values = [row.get(h, "") for h in _CSV_HEADERS]
            else:
                values = list(row)
            sheet.append_row(values)

            logger.info("Experiment registered in Google Sheets")

        except Exception as exc:
            logger.error("Google Sheets write failed: %s", exc)
            raise GoogleSheetsError(
                f"Failed to write to Google Sheets: {exc}"
            ) from exc

    # ── CSV fallback ─────────────────────────────────────────────────

    @staticmethod
    def _write_to_csv(row: dict[str, str]) -> None:
        """Append *row* to a local CSV file."""
        output_dir = OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / _CSV_FILENAME

        file_exists = csv_path.exists()

        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_HEADERS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        logger.info("Experiment registered in CSV: %s", csv_path)
