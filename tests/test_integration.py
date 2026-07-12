"""End-to-end integration tests running all 3 datasets through the pipeline."""

from pathlib import Path

import pytest

from src.config import AppConfig
from src.orchestrator.orchestrator import Orchestrator
from src.models import ExperimentReport, Recommendation


DATASETS = [
    ("dataset_01_parceiroA.csv", "Experimento A", "Parceiro A"),
    ("dataset_02_parceiroB.csv", "Experimento B", "Parceiro B"),
    ("dataset_03_parceiroC.csv", "Experimento C", "Parceiro C"),
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    """AppConfig pointing output to a temp directory."""
    return AppConfig(output_dir=tmp_path)


class TestEndToEnd:
    """Full pipeline tests for all provided datasets."""

    @pytest.mark.parametrize("csv_file,experiment,partner", DATASETS)
    def test_pipeline_completes(
        self,
        config: AppConfig,
        csv_file: str,
        experiment: str,
        partner: str,
    ) -> None:
        csv_path = PROJECT_ROOT / csv_file
        if not csv_path.exists():
            pytest.skip(f"Dataset not found: {csv_path}")

        orchestrator = Orchestrator(config)
        report = orchestrator.run(str(csv_path), experiment, partner)

        assert isinstance(report, ExperimentReport)
        assert report.experiment_name == experiment
        assert report.partner == partner
        assert report.n_variants >= 2
        assert report.decision.recommendation in Recommendation
        assert report.decision.justification != ""
        assert report.decision.winning_variant != ""
        assert report.period != ""
        assert report.timestamp != ""

    @pytest.mark.parametrize("csv_file,experiment,partner", DATASETS)
    def test_report_files_created(
        self,
        config: AppConfig,
        csv_file: str,
        experiment: str,
        partner: str,
        tmp_path: Path,
    ) -> None:
        csv_path = PROJECT_ROOT / csv_file
        if not csv_path.exists():
            pytest.skip(f"Dataset not found: {csv_path}")

        orchestrator = Orchestrator(config)
        orchestrator.run(str(csv_path), experiment, partner)

        # Check Markdown report exists
        md_files = list(tmp_path.glob("*.md"))
        assert len(md_files) >= 1

        # Check charts exist
        plot_files = list((tmp_path / "plots").glob("*.png"))
        assert len(plot_files) >= 4

    def test_dataset_03_two_groups(self, config: AppConfig) -> None:
        """Dataset 03 has exactly 2 groups — verify two-group path."""
        csv_path = PROJECT_ROOT / "dataset_03_parceiroC.csv"
        if not csv_path.exists():
            pytest.skip("Dataset not found")

        orchestrator = Orchestrator(config)
        report = orchestrator.run(str(csv_path), "Test C", "Parceiro C")

        assert report.n_variants == 2
        stats = report.decision.statistical_summary.tests
        assert "primary_test" in stats

    def test_dataset_01_three_groups(self, config: AppConfig) -> None:
        """Dataset 01 has 3 groups — verify multi-group path."""
        csv_path = PROJECT_ROOT / "dataset_01_parceiroA.csv"
        if not csv_path.exists():
            pytest.skip("Dataset not found")

        orchestrator = Orchestrator(config)
        report = orchestrator.run(str(csv_path), "Test A", "Parceiro A")

        assert report.n_variants == 3
        stats = report.decision.statistical_summary.tests
        assert "omnibus_test" in stats

    def test_csv_log_created(self, config: AppConfig, tmp_path: Path) -> None:
        """Verify experiment log CSV is created when no Sheets credentials."""
        csv_path = PROJECT_ROOT / "dataset_03_parceiroC.csv"
        if not csv_path.exists():
            pytest.skip("Dataset not found")

        orchestrator = Orchestrator(config)
        orchestrator.run(str(csv_path), "Test CSV", "Parceiro C")

        # The CSV fallback writes to constants.OUTPUT_DIR, not tmp_path.
        # Check the file exists in the default output dir.
        from src.constants import OUTPUT_DIR
        log_path = OUTPUT_DIR / "experiment_log.csv"
        assert log_path.exists()
