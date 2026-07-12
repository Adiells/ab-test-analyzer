"""AI-native conversational interface for the AB Test Analyzer.

A thin interactive layer that:
1. Receives natural-language requests.
2. Uses an LLM (Google Gemini) with conversation memory to interpret
   intent, match datasets, and maintain session context.
3. Invokes the existing orchestrator.
4. Displays a concise executive summary.
5. Remains in conversation until the user exits.

No business logic, statistics, metrics, or report generation lives here.
All analytical computation is delegated to the existing pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from src.config import AppConfig, LLMConfig
from src.constants import PROJECT_ROOT
from src.models import ExperimentReport, Recommendation
from src.orchestrator.orchestrator import Orchestrator

# ── Constants ────────────────────────────────────────────────────────

INPUT_DIR = PROJECT_ROOT / "input"

_RECOMMENDATION_LABELS: dict[Recommendation, str] = {
    Recommendation.SCALE_TREATMENT: "Escalar variante vencedora para 100% do tráfego",
    Recommendation.KEEP_CONTROL: "Manter experimento atual (sem alterações)",
    Recommendation.COLLECT_MORE_DATA: "Coletar mais dados antes de decidir",
    Recommendation.INCONCLUSIVE: "Inconclusivo — evidências insuficientes",
}

# ── Style helpers ────────────────────────────────────────────────────

_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"
_RULE = "─" * 50


def _supports_color() -> bool:
    """Check if the terminal supports ANSI colors."""
    if os.getenv("NO_COLOR"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOR = _supports_color()


def _c(code: str, text: str) -> str:
    """Apply ANSI color code if supported."""
    return f"{code}{text}{_RESET}" if _COLOR else text


def _rule() -> str:
    return _c(_DIM, _RULE)


def _label(key: str, value: str) -> str:
    return f"  {_c(_DIM, key + ':')}  {_c(_BOLD, value)}"


# ── Dataset discovery ────────────────────────────────────────────────

def _discover_datasets(search_dir: Path | None = None) -> list[Path]:
    """Find all CSV files in the input directory."""
    directory = search_dir or INPUT_DIR
    if not directory.exists():
        return []
    return sorted(directory.glob("*.csv"))


# ── LLM conversation engine ─────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are the AI assistant for an A/B Test Analyzer platform.
You help users analyze A/B experiments (such as cashback tests) by
understanding their natural-language requests and selecting the correct
dataset to analyze.

CAPABILITIES:
- Identify which dataset the user wants to analyze.
- Ask clarifying questions when the request is ambiguous.
- Remember previous analyses within the same session.
- Understand references like "the first one", "the previous experiment",
  "that dataset", "the second one", "partner B's experiment", etc.

AVAILABLE DATASETS will be provided in each message as context.
Previous analysis results will also be provided when available.

YOUR RESPONSE must ALWAYS be a valid JSON object with this schema:

{
  "intent": "analyze" | "exit" | "greeting" | "help" | "unclear",
  "dataset": "<exact filename from the available list>" | null,
  "ambiguous": ["<filename>", ...] | null,
  "message": "<friendly conversational message to display>"
}

RULES:
- "intent": the user's intent.
  - "analyze": the user wants to run an analysis on a dataset.
  - "exit": the user wants to quit (e.g. "exit", "quit", "sair", "bye", "tchau", "q").
  - "greeting": the user is saying hi or asking what you can do.
  - "help": the user asks for help or available options.
  - "unclear": you cannot determine what the user wants.

- "dataset": the EXACT filename from the available list. Set only when
  you are confident about which dataset the user means. null otherwise.

- "ambiguous": a list of candidate filenames when multiple datasets
  could match the user's request. null when not applicable.

- "message": a short, friendly, conversational message written in the
  SAME LANGUAGE the user used. This is what the user will see.
  - For "analyze" with a match: confirm which dataset you will analyze.
  - For "analyze" with ambiguity: ask which one they mean, listing options.
  - For "analyze" with no match: say you couldn't find it and suggest options.
  - For "greeting"/"help": explain what you can do and list available datasets.
  - For "exit": say goodbye warmly.

DATASET MATCHING — use reasoning, not keyword matching:
- "partner A" = "parceiro A" = "parceiroA"
- "dataset 1" = "dataset_01" = "first dataset" = "primeiro dataset"
- "the second one" = refers to the second dataset in the available list
- "the previous experiment" = the last dataset that was analyzed (from session history)
- "that dataset" = the most recently mentioned dataset
- Be bilingual: understand both English and Portuguese.

Respond ONLY with the JSON object. No markdown, no explanation.
"""


def _build_context_block(datasets: list[Path], analysis_history: list[dict]) -> str:
    """Build context about available datasets and past analyses."""
    lines = ["[CONTEXT]"]
    lines.append("")
    lines.append("Available datasets:")
    for i, ds in enumerate(datasets, 1):
        lines.append(f"  {i}. {ds.name}")

    if analysis_history:
        lines.append("")
        lines.append("Previous analyses in this session:")
        for i, entry in enumerate(analysis_history, 1):
            lines.append(
                f"  {i}. {entry['dataset']} → "
                f"recommendation: {entry['recommendation']}, "
                f"winning variant: {entry['winning_variant']}"
            )

    lines.append("")
    return "\n".join(lines)


class _ConversationEngine:
    """Manages the LLM conversation with session memory.

    Maintains a rolling conversation history so the LLM can resolve
    contextual references like "the previous experiment" or "the second one".
    """

    def __init__(self, llm_config: LLMConfig) -> None:
        self._config = llm_config
        self._history: list[dict[str, str]] = []
        self._analysis_history: list[dict] = []
        self._max_history_turns = 20  # Keep last N user/assistant pairs.

    def record_analysis(self, dataset_name: str, report: ExperimentReport) -> None:
        """Record a completed analysis for session context."""
        d = report.decision
        rec_label = _RECOMMENDATION_LABELS.get(
            d.recommendation, d.recommendation.value,
        )
        self._analysis_history.append({
            "dataset": dataset_name,
            "partner": report.partner,
            "recommendation": rec_label,
            "winning_variant": d.winning_variant or "N/A",
            "significance": report.statistical_significance,
            "confidence": f"{d.confidence:.1%}",
            "n_variants": report.n_variants,
        })

        # Also inject a summary into conversation history so the LLM
        # knows what happened.
        summary = (
            f"[ANALYSIS COMPLETED] Dataset: {dataset_name} | "
            f"Partner: {report.partner} | "
            f"Recommendation: {rec_label} | "
            f"Winning variant: {d.winning_variant or 'N/A'} | "
            f"Significance: {'Yes' if report.statistical_significance else 'No'}"
        )
        self._history.append({"role": "assistant", "content": summary})

    def interpret(self, user_message: str, datasets: list[Path]) -> dict:
        """Send the user's message to the LLM with full conversation context.

        Returns a dict with keys: intent, dataset, ambiguous, message.
        """
        context = _build_context_block(datasets, self._analysis_history)
        full_user_message = f"{context}\n[USER MESSAGE]\n{user_message}"

        # Add to history.
        self._history.append({"role": "user", "content": full_user_message})
        self._trim_history()

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self._config.api_key)

            # Build contents from conversation history.
            contents = []
            for entry in self._history:
                contents.append(
                    types.Content(
                        role=("user" if entry["role"] == "user" else "model"),
                        parts=[types.Part.from_text(text=entry["content"])],
                    )
                )

            response = client.models.generate_content(
                model=self._config.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )

            raw = response.text.strip()

            # Strip markdown code fences if the model wraps in ```json```.
            if raw.startswith("```"):
                lines = raw.split("\n")
                lines = [ln for ln in lines if not ln.strip().startswith("```")]
                raw = "\n".join(lines)

            result = json.loads(raw)

            # Record the assistant's response in history.
            self._history.append({"role": "assistant", "content": raw})

            return result

        except Exception as exc:
            logging.getLogger(__name__).debug("LLM call failed: %s", exc)
            # Remove the failed user message from history.
            if self._history and self._history[-1]["role"] == "user":
                self._history.pop()
            return {
                "intent": "unclear",
                "dataset": None,
                "ambiguous": None,
                "message": None,
            }

    def _trim_history(self) -> None:
        """Keep conversation history within bounds."""
        max_entries = self._max_history_turns * 2  # user + assistant pairs
        if len(self._history) > max_entries:
            self._history = self._history[-max_entries:]


# ── Keyword-based fallback (no API key) ──────────────────────────────

def _parse_intent_fallback(user_message: str, datasets: list[Path]) -> dict:
    """Simple keyword-based fallback when no LLM API key is available."""
    text = user_message.lower().strip()

    exit_words = {"exit", "quit", "bye", "sair", "q", "tchau", "adeus"}
    if text in exit_words:
        return {"intent": "exit", "dataset": None, "ambiguous": None, "message": "Goodbye!"}

    greeting_words = {"hi", "hello", "hey", "oi", "olá", "ola", "help", "ajuda"}
    if text in greeting_words:
        return {"intent": "greeting", "dataset": None, "ambiguous": None, "message": None}

    if datasets:
        matches = []
        for ds in datasets:
            name_lower = ds.name.lower()
            tokens = text.replace("_", " ").replace("-", " ").split()
            for tok in tokens:
                if len(tok) > 1 and tok in name_lower:
                    matches.append(ds)
                    break

        if len(matches) == 1:
            return {
                "intent": "analyze",
                "dataset": matches[0].name,
                "ambiguous": None,
                "message": f"Analyzing {matches[0].name}...",
            }
        if len(matches) > 1:
            return {
                "intent": "analyze",
                "dataset": None,
                "ambiguous": [m.name for m in matches],
                "message": "I found multiple possible datasets.",
            }

    return {"intent": "unclear", "dataset": None, "ambiguous": None, "message": None}


# ── Derive experiment metadata from filename ────────────────────────

def _derive_experiment_name(path: Path) -> str:
    return path.stem.replace("_", " ").title()


def _derive_partner(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")
    partner_parts = [p for p in parts if p.lower().startswith("parceiro")]
    if partner_parts:
        return " ".join(partner_parts).title()
    return parts[-1].title() if parts else stem


# ── Pipeline execution ───────────────────────────────────────────────

def _run_analysis(dataset_path: Path) -> ExperimentReport | None:
    """Invoke the existing orchestrator and return the report.

    Returns None if the pipeline fails.
    """
    experiment_name = _derive_experiment_name(dataset_path)
    partner = _derive_partner(dataset_path)

    print()
    print(_c(_DIM, f"  ⏳ Running analysis on {dataset_path.name}..."))
    print()

    # Suppress verbose pipeline logs from polluting the chat UI.
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(logging.WARNING)

    try:
        config = AppConfig()
        orchestrator = Orchestrator(config)
        report = orchestrator.run(
            file_path=str(dataset_path),
            experiment_name=experiment_name,
            partner=partner,
        )
        return report
    except Exception as exc:
        print(_c(_RED, f"  ✗ Analysis failed: {exc}"))
        print()
        return None
    finally:
        root_logger.setLevel(previous_level)


# ── Summary display ──────────────────────────────────────────────────

def _display_summary(report: ExperimentReport) -> None:
    """Print a concise executive summary of the analysis."""
    d = report.decision
    rec_label = _RECOMMENDATION_LABELS.get(
        d.recommendation, d.recommendation.value,
    )

    sig_text = _c(_GREEN, "Yes") if report.statistical_significance else _c(_YELLOW, "No")

    print(_rule())
    print()
    print(_c(_GREEN, "  ✓ Experiment completed successfully."))
    print()
    print(_label("Partner", report.partner))
    print(_label("Period", report.period))
    print(_label("Variants", str(report.n_variants)))
    print(_label("Winning Variant", d.winning_variant or "N/A"))
    print(_label("Statistical Significance", sig_text))
    print(_label("Confidence", f"{d.confidence:.1%}"))
    print()
    print(f"  {_c(_BOLD, 'Recommendation:')}  {rec_label}")
    print()
    print(_rule())
    print()

    safe_name = report.experiment_name.replace(" ", "_").lower()
    print(_c(_DIM, f"  Report saved to:       output/{safe_name}_report.md"))
    print(_c(_DIM, f"  Plots saved to:        output/plots/"))
    print(_c(_DIM, f"  Experiment registered:  Google Sheets / CSV fallback"))
    print()
    print(_rule())
    print()


# ── Display helpers ──────────────────────────────────────────────────

def _show_datasets(datasets: list[Path]) -> None:
    """List available datasets."""
    if datasets:
        print()
        print(_c(_DIM, "  Available datasets:"))
        print()
        for ds in datasets:
            print(f"    • {ds.name}")
        print()


def _resolve_dataset(name: str, datasets: list[Path]) -> Path | None:
    """Resolve a filename string to a Path from the datasets list."""
    for ds in datasets:
        if ds.name == name:
            return ds
    return None


# ── Main conversation loop ───────────────────────────────────────────

def _print_banner() -> None:
    print()
    print(_rule())
    print(f"  {_c(_BOLD, 'A/B Test Analyzer')}")
    print(f"  {_c(_DIM, 'AI-Native Assistant')}")
    print(_rule())
    print()
    print("  How can I help you?")
    print()


def chat() -> None:
    """Start the interactive conversational session."""
    from src.logging_config import setup_logging
    setup_logging(level=logging.WARNING)

    config = AppConfig()
    llm_config = config.llm
    use_llm = bool(llm_config.api_key)

    engine: _ConversationEngine | None = None
    if use_llm:
        engine = _ConversationEngine(llm_config)
    else:
        logging.getLogger(__name__).info(
            "No LLM_API_KEY — using keyword fallback for intent parsing"
        )

    datasets = _discover_datasets()
    pending_disambiguation: list[str] | None = None

    _print_banner()

    while True:
        try:
            user_input = input(_c(_CYAN, "  > ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print(_c(_DIM, "  Goodbye!"))
            print()
            break

        if not user_input:
            continue

        # ── Pending disambiguation: user is picking from a list ──
        if pending_disambiguation is not None:
            if user_input.isdigit():
                idx = int(user_input)
                if 1 <= idx <= len(pending_disambiguation):
                    filename = pending_disambiguation[idx - 1]
                    selected = _resolve_dataset(filename, datasets)
                    pending_disambiguation = None
                    if selected:
                        report = _run_analysis(selected)
                        if report is not None:
                            _display_summary(report)
                            if engine:
                                engine.record_analysis(selected.name, report)
                        print("  Anything else?")
                        print()
                    continue
                else:
                    print(_c(_YELLOW, f"  Please type a number between 1 and {len(pending_disambiguation)}."))
                    continue

            # Not a number — pass to the LLM to re-interpret (might be
            # a dataset name or a completely new request).
            pending_disambiguation = None

        # ── Refresh dataset list ─────────────────────────────────
        datasets = _discover_datasets()

        # ── Parse intent via LLM (with memory) or fallback ───────
        if engine:
            result = engine.interpret(user_input, datasets)
        else:
            result = _parse_intent_fallback(user_input, datasets)

        intent = result.get("intent", "unclear")
        dataset_name = result.get("dataset")
        ambiguous = result.get("ambiguous")
        message = result.get("message")

        # ── Exit ─────────────────────────────────────────────────
        if intent == "exit":
            print()
            print(_c(_DIM, f"  {message or 'Goodbye!'}"))
            print()
            break

        # ── Greeting / help ──────────────────────────────────────
        if intent in ("greeting", "help"):
            if message:
                print()
                print(f"  {message}")
            _show_datasets(datasets)
            continue

        # ── Analyze with a single match ──────────────────────────
        if intent == "analyze" and dataset_name:
            selected = _resolve_dataset(dataset_name, datasets)
            if selected:
                if message:
                    print()
                    print(f"  {message}")
                report = _run_analysis(selected)
                if report is not None:
                    _display_summary(report)
                    if engine:
                        engine.record_analysis(selected.name, report)
                print("  Anything else?")
                print()
                continue
            else:
                print()
                print(_c(_YELLOW, f"  Dataset '{dataset_name}' not found."))
                _show_datasets(datasets)
                continue

        # ── Analyze with ambiguity ───────────────────────────────
        if intent == "analyze" and ambiguous:
            pending_disambiguation = ambiguous
            if message:
                print()
                print(f"  {message}")
                print()
                for i, name in enumerate(ambiguous, 1):
                    print(f"    {_c(_BOLD, str(i))}. {name}")
                print()
            continue

        # ── Analyze with no match ────────────────────────────────
        if intent == "analyze":
            print()
            print(_c(_YELLOW, f"  {message or 'I could not find a matching dataset.'}"))
            _show_datasets(datasets)
            continue

        # ── Unclear ──────────────────────────────────────────────
        if message:
            print()
            print(f"  {message}")
            print()
        else:
            print()
            print(_c(_YELLOW, "  I'm not sure what you mean."))
            print("  Try something like: \"analyze dataset 1\" or \"analyze partner A\"")
            _show_datasets(datasets)


# ── Entrypoint ───────────────────────────────────────────────────────

def main() -> None:
    """Console script entrypoint for ``teste-ab``."""
    chat()


if __name__ == "__main__":
    main()
