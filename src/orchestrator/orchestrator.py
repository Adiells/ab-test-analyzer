"""Pipeline orchestrator.

Coordinates the end-to-end experiment analysis by invoking each
pipeline stage in sequence and passing intermediate results forward.

The orchestrator owns the execution flow but delegates all logic
to the individual services.
"""

from __future__ import annotations

from pathlib import Path

from src.config import AppConfig
from src.core.decision.decision import DecisionService
from src.core.ingestion.ingestion import IngestionService
from src.core.metrics.metrics import MetricsService
from src.core.normalization.normalization import NormalizationService
from src.core.preprocessing.preprocessing import PreprocessingService
from src.core.statistics.statistics import StatisticsService
from src.core.validation.validation import ValidationService
from src.integrations.google_sheets.sheets_service import GoogleSheetsService
from src.llm.llm_service import LLMService
from src.logging_config import get_logger
from src.models import ExperimentReport
from src.reporting.report_service import ReportService

logger = get_logger(__name__)


class Orchestrator:
    """Runs the full A/B test analysis pipeline.

    Each public method corresponds to a stage in the pipeline.
    The ``run`` method executes them in order.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config

        # Instantiate services
        self._ingestion = IngestionService()
        self._normalization = NormalizationService()
        self._validation = ValidationService()
        self._preprocessing = PreprocessingService()
        self._metrics = MetricsService()
        self._statistics = StatisticsService(config.statistics)
        self._decision = DecisionService()
        self._llm = LLMService(config.llm)
        self._report = ReportService(config)
        self._sheets = GoogleSheetsService(config.google_sheets)

    def run(
        self,
        file_path: str | Path,
        experiment_name: str,
        partner: str,
    ) -> ExperimentReport:
        """Execute the full analysis pipeline.

        Args:
            file_path: Path to the experiment CSV dataset.
            experiment_name: Human-readable experiment identifier.
            partner: Partner associated with the experiment.

        Returns:
            The completed ``ExperimentReport``.
        """
        logger.info("Starting analysis pipeline for '%s'", experiment_name)

        # 1. Ingest
        logger.info("Stage 1/11: Ingestion")
        dataset = self._ingestion.load(file_path)

        # 2. Normalize
        logger.info("Stage 2/11: Normalization")
        normalized_data = self._normalization.normalize(dataset.data)

        # 3. Validate
        logger.info("Stage 3/11: Validation")
        validation = self._validation.validate(normalized_data)
        if validation.warnings:
            logger.warning("Validation warnings: %s", validation.warnings)
        if not validation.is_valid:
            logger.error("Validation failed: %s", validation.errors)
            raise ValueError(
                f"Dataset validation failed with errors: {validation.errors}"
            )

        # 4. Preprocess
        logger.info("Stage 4/11: Preprocessing")
        processed_data = self._preprocessing.preprocess(normalized_data)

        # 5. Compute metrics
        logger.info("Stage 5/11: Metrics")
        metrics = self._metrics.compute(processed_data)

        # 6. Statistical analysis
        logger.info("Stage 6/11: Statistical analysis")
        statistics = self._statistics.analyze(processed_data, metrics)

        # 7. Decision
        logger.info("Stage 7/11: Decision engine")
        decision = self._decision.decide(metrics, statistics)

        # 8. LLM narrative
        logger.info("Stage 8/11: LLM narrative")
        narrative = self._llm.generate_narrative(decision)

        # 9. Build report
        logger.info("Stage 9/11: Build report")
        report = self._report.build_report(
            experiment_name=experiment_name,
            partner=partner,
            decision=decision,
            narrative=narrative,
        )

        # 10. Save report
        logger.info("Stage 10/11: Save report")
        self._report.save(report)

        # 11. Register in Google Sheets
        logger.info("Stage 11/11: Register experiment")
        self._sheets.register_experiment(report)

        logger.info("Pipeline completed for '%s'", experiment_name)
        return report

if __name__ == "__main__":
    orchestrator = Orchestrator(AppConfig())
    report = orchestrator.run(
        "dataset_01_parceiroA.csv",
        "dataset_01_parceiroA",
        "Parceiro A",
    )
    print(report)