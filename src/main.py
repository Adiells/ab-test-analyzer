"""AB Test Analyzer — entry point.

Usage:
    python -m src.main <dataset_path> <experiment_name> <partner>
"""

from __future__ import annotations

import argparse
import sys

from src.config import AppConfig
from src.logging_config import get_logger, setup_logging
from src.orchestrator.orchestrator import Orchestrator

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Native A/B Test Analyzer",
    )
    parser.add_argument(
        "dataset",
        help="Path to the experiment CSV dataset.",
    )
    parser.add_argument(
        "experiment",
        help="Human-readable experiment name.",
    )
    parser.add_argument(
        "partner",
        help="Partner associated with the experiment.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Application entry point."""
    setup_logging()

    args = parse_args(argv)
    config = AppConfig()

    logger.info(
        "AB Test Analyzer — experiment=%s partner=%s",
        args.experiment,
        args.partner,
    )

    orchestrator = Orchestrator(config)

    try:
        report = orchestrator.run(
            file_path=args.dataset,
            experiment_name=args.experiment,
            partner=args.partner,
        )
        logger.info("Analysis complete. Recommendation: %s", report.decision.recommendation.value)
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
