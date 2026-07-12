"""LLM client service.

Interacts with a language model to generate executive narratives
from structured experiment results.  The LLM is never responsible
for calculations — only for interpreting and communicating results.

Provider-agnostic: uses the OpenAI-compatible chat completions API,
which is supported by OpenAI, Azure OpenAI, Anthropic adapters,
local models (Ollama, vLLM), and others.

Falls back to a deterministic structured narrative when no API key
is configured.
"""

from __future__ import annotations

from src.config import LLMConfig
from src.exceptions import LLMError
from src.logging_config import get_logger
from src.models import Decision, Recommendation

logger = get_logger(__name__)

_RECOMMENDATION_LABELS: dict[Recommendation, str] = {
    Recommendation.SCALE_TREATMENT: "Escalar variante vencedora para 100% do tráfego",
    Recommendation.KEEP_CONTROL: "Manter experimento atual (sem alterações)",
    Recommendation.COLLECT_MORE_DATA: "Coletar mais dados antes de decidir",
    Recommendation.INCONCLUSIVE: "Inconclusivo — evidências insuficientes",
}

_SYSTEM_PROMPT = """\
You are a senior data analyst writing an executive summary of an A/B test.

Rules:
- Explain the results in clear, non-technical language.
- Do NOT recalculate any metrics — use only the data provided.
- Do NOT make business decisions — the recommendation has already been made.
- Focus on what the numbers mean for the business.
- Be concise: 3-5 paragraphs maximum.
- Use Markdown formatting.
- Write the entire response strictly in Portuguese.
"""


class LLMService:
    """Generates natural-language narratives via an LLM."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    def generate_narrative(self, decision: Decision) -> str:
        """Produce an executive narrative for the experiment results.

        If an API key is configured, calls the LLM.  Otherwise,
        generates a deterministic structured narrative.

        Args:
            decision: The structured decision output from the pipeline.

        Returns:
            A Markdown-formatted executive summary string.
        """
        if not self._config.api_key:
            logger.info(
                "No LLM API key configured — using deterministic narrative"
            )
            return self._build_fallback_narrative(decision)

        logger.info(
            "Generating narrative with model=%s", self._config.model_name,
        )
        return self._call_llm(decision)

    # ── LLM call ─────────────────────────────────────────────────────

    def _call_llm(self, decision: Decision) -> str:
        """Send structured data to the LLM and return the response."""
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self._config.api_key)

            user_prompt = self._build_prompt(decision)

            response = client.models.generate_content(
                model=self._config.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    temperature=self._config.temperature,
                    max_output_tokens=self._config.max_tokens,
                ),
            )

            narrative = response.text or ""
            logger.info("LLM narrative generated (%d chars)", len(narrative))
            return narrative.strip()

            #TODO choose the final llm model to use.

        except ImportError:
            logger.warning("openai package not installed — using fallback")
            return self._build_fallback_narrative(decision)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise LLMError(f"LLM narrative generation failed: {exc}") from exc

    # ── Prompt construction ──────────────────────────────────────────

    @staticmethod
    def _build_prompt(decision: Decision) -> str:
        """Build a structured prompt from the Decision data."""
        stats = decision.statistical_summary.tests
        summary = decision.metrics_summary.summary
        group_names = stats.get("group_names", [])

        lines = [
            "## Experiment Results",
            "",
            f"**Recommendation:** {decision.recommendation.value}",
            f"**Winning variant:** {decision.winning_variant}",
            f"**Confidence:** {decision.confidence:.1%}",
            "",
            "### Metrics by Group",
            "",
        ]

        for group in group_names:
            data = summary.get(group, {})
            lines.append(f"**{group}:**")
            lines.append(f"- Total buyers: {data.get('total_buyers', 0):,.0f}")
            lines.append(f"- Total sales: R$ {data.get('total_sales', 0):,.2f}")
            lines.append(f"- Total commission: R$ {data.get('total_commission', 0):,.2f}")
            lines.append(f"- Total cashback: R$ {data.get('total_cashback', 0):,.2f}")
            lines.append(f"- Net revenue: R$ {data.get('total_net_revenue', 0):,.2f}")
            lines.append(f"- Net margin: {data.get('net_margin', 0):.2%}")
            lines.append("")

        lines.append("### Statistical Analysis")
        lines.append("")

        primary = stats.get("primary_test")
        if primary:
            lines.append(f"- Method: {primary.get('method')}")
            lines.append(f"- p-value: {primary.get('p_value')}")
            lines.append(f"- Significant: {'Yes' if primary.get('significant') else 'No'}")
            cd = primary.get("cohens_d", {})
            if cd:
                lines.append(f"- Effect size (Cohen's d): {cd.get('d')} ({cd.get('magnitude')})")

        omnibus = stats.get("omnibus_test")
        if omnibus:
            lines.append(f"- Method: {omnibus.get('method')}")
            lines.append(f"- p-value: {omnibus.get('p_value')}")
            lines.append(f"- Significant: {'Yes' if omnibus.get('significant') else 'No'}")

        lines.append("")
        lines.append("### Decision Justification")
        lines.append("")
        lines.append(decision.justification)

        if decision.risks:
            lines.append("")
            lines.append("### Risks")
            for risk in decision.risks:
                lines.append(f"- {risk}")

        lines.append("")
        lines.append(
            "Please explain these results in an executive-friendly narrative. "
            "Do not recalculate — only interpret the numbers provided. "
            "Write the entire output strictly in Portuguese."
        )

        return "\n".join(lines)

    # ── Fallback narrative ───────────────────────────────────────────

    @staticmethod
    def _build_fallback_narrative(decision: Decision) -> str:
        """Generate a deterministic narrative when no LLM is available."""
        stats = decision.statistical_summary.tests
        summary = decision.metrics_summary.summary
        group_names = stats.get("group_names", [])

        rec_label = _RECOMMENDATION_LABELS.get(
            decision.recommendation, decision.recommendation.value,
        )

        lines = [
            "### Análise do Experimento",
            "",
            f"Este experimento comparou **{len(group_names)} variante(s)**: "
            f"{', '.join(group_names)}.",
            "",
        ]

        # Metrics highlights.
        if group_names:
            best = decision.winning_variant or group_names[0]
            best_data = summary.get(best, {})
            lines.append(
                f"A variante de melhor desempenho foi **{best}**, com uma receita "
                f"líquida total de **R$ {best_data.get('total_net_revenue', 0):,.2f}** "
                f"em **{best_data.get('total_buyers', 0):,.0f} compradores** e uma "
                f"margem líquida de **{best_data.get('net_margin', 0):.2%}**."
            )
            lines.append("")

        # Statistical highlights.
        primary = stats.get("primary_test")
        omnibus = stats.get("omnibus_test")
        test_info = primary or omnibus
        if test_info:
            method = test_info.get("method", "teste de hipótese")
            p_value = test_info.get("p_value", "N/A")
            significant = test_info.get("significant", False)
            sig_word = "estatisticamente significativas" if significant else "não estatisticamente significativas"
            lines.append(
                f"O {method} resultou em um p-valor de **{p_value}**, indicando "
                f"que as diferenças observadas são **{sig_word}**."
            )
            lines.append("")

        # Decision.
        lines.append(f"**Recomendação:** {rec_label}.")
        lines.append("")
        lines.append(decision.justification)

        return "\n".join(lines)
