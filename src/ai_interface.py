"""AI-Native interface for the AB Test Analyzer.

This module exposes the analysis pipeline as **structured tool definitions**
that any LLM or AI assistant can invoke through function-calling, tool-use,
or natural-language orchestration.

Supported workflows
-------------------
1. **Direct Python call** – import ``analyze_experiment`` and call it.
2. **Function-calling / tool-use** – use ``TOOLS`` to register the schema
   with any OpenAI-compatible, Gemini, or Anthropic function-calling API.
3. **CLI one-liner** – ``python -m src.ai_interface <dataset_path>``

The only required parameter is ``dataset_path``.  Everything else is
inferred automatically from the dataset file name when omitted.

Examples of natural-language requests that map to this interface
---------------------------------------------------------------
* "Analyze the experiment in input/dataset_01_parceiroA.csv"
* "Run the cashback experiment for Partner B and register the results"
* "Analyze dataset_03_parceiroC.csv and tell me which variant should
   be rolled out to 100% of traffic"
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from enum import Enum
from pathlib import Path

from src.config import AppConfig
from src.logging_config import get_logger, setup_logging
from src.orchestrator.orchestrator import Orchestrator

logger = get_logger(__name__)


# ── Tool / function-calling schema ───────────────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "analyze_experiment",
            "description": (
                "Run the full A/B test analysis pipeline on a CSV dataset. "
                "Validates the data, computes business metrics, runs "
                "statistical hypothesis tests, generates an AI executive "
                "report, saves outputs (Markdown report + charts) to the "
                "'output/' directory, and registers the experiment in "
                "Google Sheets (or a local CSV fallback). "
                "Returns the complete structured analysis including the "
                "recommendation on which variant to scale."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": (
                            "Path to the experiment CSV file. "
                            "Examples: 'input/dataset_01_parceiroA.csv', "
                            "'data/july_cashback_test.csv'. "
                            "Relative paths are resolved from the project root."
                        ),
                    },
                    "experiment_name": {
                        "type": "string",
                        "description": (
                            "Human-readable name for the experiment. "
                            "If omitted, derived automatically from the "
                            "file name (e.g. 'dataset_01_parceiroA')."
                        ),
                    },
                    "partner": {
                        "type": "string",
                        "description": (
                            "Partner or business unit associated with "
                            "the experiment. If omitted, inferred from "
                            "the file name."
                        ),
                    },
                },
                "required": ["dataset_path"],
            },
        },
    },
]
"""OpenAI / Gemini / Anthropic compatible tool definitions.

Register this list with your AI assistant's tool-use system so it can
invoke ``analyze_experiment`` via function-calling.
"""


# ── Prompt templates ─────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an AI assistant integrated with the **AB Test Analyzer** platform.

You have access to the tool `analyze_experiment` which runs a complete
A/B test analysis pipeline.  When the user asks to analyze an experiment,
call this tool with the dataset path.

The tool will:
1. Validate the CSV dataset.
2. Compute business metrics (revenue, cashback, margins).
3. Run statistical hypothesis tests.
4. Generate an executive report using AI.
5. Save the report and charts to the `output/` directory.
6. Register the experiment in Google Sheets (or local CSV).

After receiving the tool result, summarize the key findings in clear,
business-friendly language.  Always include:
- The recommendation (scale winner / keep current / collect more data)
- The winning variant (if any)
- Statistical significance (yes/no)
- Key metrics comparison

Respond in the same language the user used.
"""

EXAMPLE_PROMPTS: list[dict[str, str]] = [
    {
        "user": "Analyze the A/B experiment located at input/dataset_01_parceiroA.csv and register the results.",
        "tool_call": '{"dataset_path": "input/dataset_01_parceiroA.csv"}',
    },
    {
        "user": "Analyze dataset_02_parceiroB.csv",
        "tool_call": '{"dataset_path": "input/dataset_02_parceiroB.csv"}',
    },
    {
        "user": "Run the cashback experiment for Partner C and tell me which variant wins.",
        "tool_call": '{"dataset_path": "input/dataset_03_parceiroC.csv", "partner": "Partner C"}',
    },
    {
        "user": "Analyze the latest experiment and generate the executive report.",
        "tool_call": '{"dataset_path": "input/dataset_01_parceiroA.csv"}',
    },
    {
        "user": "Analise o experimento de cashback do Parceiro A e me diga qual variante escalar.",
        "tool_call": '{"dataset_path": "input/dataset_01_parceiroA.csv", "partner": "Parceiro A"}',
    },
]
"""Few-shot examples mapping natural-language requests to tool calls."""


# ── Helper: serialise dataclass to JSON-safe dict ────────────────────

def _serialise(obj: object) -> object:
    """Recursively convert dataclasses, enums and paths to JSON-safe types."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialise(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    return obj


# ── Main entry point ─────────────────────────────────────────────────

def analyze_experiment(
    dataset_path: str,
    experiment_name: str | None = None,
    partner: str | None = None,
) -> dict:
    """Run the full A/B test analysis pipeline.

    This is the single function an AI assistant needs to call.
    Only ``dataset_path`` is required — everything else is inferred.

    Args:
        dataset_path: Path to the experiment CSV file.
        experiment_name: Human-readable name (auto-derived if omitted).
        partner: Partner name (auto-derived if omitted).

    Returns:
        A JSON-serialisable dict with the full experiment report,
        including recommendation, metrics, statistics, and narrative.
    """
    setup_logging()

    # ── Derive defaults from file name ───────────────────────────
    path = Path(dataset_path)
    stem = path.stem  # e.g. "dataset_01_parceiroA"

    if not experiment_name:
        experiment_name = stem.replace("_", " ").title()

    if not partner:
        # Attempt to extract partner from common naming patterns.
        parts = stem.split("_")
        partner_parts = [p for p in parts if p.lower().startswith("parceiro")]
        partner = " ".join(partner_parts).title() if partner_parts else stem.split("_")[-1].title()

    logger.info(
        "AI Interface — analyze_experiment(dataset=%s, experiment=%s, partner=%s)",
        dataset_path,
        experiment_name,
        partner,
    )

    # ── Run the pipeline ─────────────────────────────────────────
    config = AppConfig()
    orchestrator = Orchestrator(config)

    report = orchestrator.run(
        file_path=dataset_path,
        experiment_name=experiment_name,
        partner=partner,
    )

    # ── Build structured response ────────────────────────────────
    result = {
        "status": "success",
        "experiment_name": report.experiment_name,
        "partner": report.partner,
        "period": report.period,
        "n_variants": report.n_variants,
        "statistical_significance": report.statistical_significance,
        "recommendation": report.decision.recommendation.value,
        "winning_variant": report.decision.winning_variant or None,
        "confidence": report.decision.confidence,
        "justification": report.decision.justification,
        "risks": report.decision.risks,
        "metrics": _serialise(report.decision.metrics_summary.summary),
        "statistics": _serialise(report.decision.statistical_summary.tests),
        "narrative": report.narrative,
        "timestamp": report.timestamp,
        "output_files": {
            "report": f"output/{report.experiment_name.replace(' ', '_').lower()}_report.md",
            "plots_dir": "output/plots/",
        },
    }

    return result


# ── CLI entry point ──────────────────────────────────────────────────

def main() -> None:
    """Minimal CLI wrapper — accepts just a dataset path.

    Usage::

        python -m src.ai_interface input/dataset_01_parceiroA.csv
        python -m src.ai_interface input/dataset_02_parceiroB.csv "My Experiment" "Partner B"
    """
    if len(sys.argv) < 2:
        print("Usage: python -m src.ai_interface <dataset_path> [experiment_name] [partner]")
        sys.exit(1)

    dataset_path = sys.argv[1]
    experiment_name = sys.argv[2] if len(sys.argv) > 2 else None
    partner = sys.argv[3] if len(sys.argv) > 3 else None

    result = analyze_experiment(dataset_path, experiment_name, partner)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
