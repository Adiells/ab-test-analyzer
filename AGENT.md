# AGENT.md — Instructions for AI Assistants

This file tells AI tools (Claude Code, Cursor, ChatGPT, Gemini, GitHub Copilot, GPT Custom Assistants, etc.) how to use the **AB Test Analyzer**.

## What this project does

Analyzes A/B experiments (such as cashback tests) from CSV datasets. The pipeline validates data, computes business metrics, runs statistical hypothesis tests, generates an AI-powered executive report, and registers results in Google Sheets or a local CSV log.

## Primary interaction: Conversational AI Assistant

The project includes a built-in LLM-powered conversational assistant:

```bash
teste-ab
```

This starts an interactive session where users describe their analysis in natural language. The assistant uses **Google Gemini** to:
- Understand the user's intent
- Automatically discover and select the correct dataset
- Maintain conversation context across multiple turns
- Ask clarification questions when ambiguous

**No filenames, paths, or commands are required.** Users simply describe what they want.

### Example session

```text
> analyze partner A
→ runs analysis on dataset_01_parceiroA.csv

> now analyze the second one
→ runs analysis on dataset_02_parceiroB.csv (contextual reference)

> exit
```

## Programmatic interface (for external AI tool integration)

For invoking the pipeline from another AI assistant:

```python
from src.ai_interface import analyze_experiment

result = analyze_experiment("input/dataset_01_parceiroA.csv")
```

### Parameters

| Parameter         | Required | Description                                      |
|-------------------|----------|--------------------------------------------------|
| `dataset_path`    | **Yes**  | Path to the CSV file with experiment data        |
| `experiment_name` | No       | Human-readable name (auto-derived from filename) |
| `partner`         | No       | Partner name (auto-derived from filename)         |

### Return value

A JSON dict with:
- `status`: `"success"` or error info
- `recommendation`: one of `scale_treatment`, `keep_control`, `collect_more_data`, `inconclusive`
- `winning_variant`: which variant won (or null)
- `statistical_significance`: boolean
- `confidence`: float (0–1)
- `metrics`: per-group business metrics
- `narrative`: AI-generated executive summary
- `output_files`: paths to saved report and charts

## Tool schema for function-calling

The structured tool definition is available at:

```python
from src.ai_interface import TOOLS
```

This returns an OpenAI/Gemini/Anthropic compatible tool schema list ready to be registered with any function-calling API.

## System prompt

A ready-to-use system prompt is available at:

```python
from src.ai_interface import SYSTEM_PROMPT
```

## Available datasets

Datasets are located in the `input/` directory:
- `input/dataset_01_parceiroA.csv`
- `input/dataset_02_parceiroB.csv`
- `input/dataset_03_parceiroC.csv`

## Example natural-language requests

All of the following are understood by the assistant:

- "analyze partner A"
- "run the cashback experiment for partner C"
- "analyze the first dataset"
- "analise o experimento do parceiro B"
- "execute the A/B test for dataset 02"
- "now analyze the second one" (contextual reference)
- "analyze the previous experiment" (session memory)

## No code changes needed

To analyze a **new experiment**, only the dataset file changes. No source code modifications are required. Place the CSV in the `input/` directory and the assistant will discover it automatically.

## Output

Results are saved to:
- `output/<experiment_name>_report.md` — full Markdown report
- `output/plots/` — comparison charts (PNG)
- `output/experiment_log.csv` — consolidated log (or Google Sheets)
