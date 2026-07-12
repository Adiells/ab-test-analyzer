"""Report generation service.

Assembles the final executive report from structured analysis
results and the LLM-generated narrative.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI use.
import matplotlib.pyplot as plt

from src.config import AppConfig
from src.exceptions import ReportGenerationError
from src.logging_config import get_logger
from src.models import Decision, ExperimentReport, Recommendation

logger = get_logger(__name__)

# Human-readable labels for enum values.
_RECOMMENDATION_LABELS: dict[Recommendation, str] = {
    Recommendation.SCALE_TREATMENT: "Escalar variante vencedora para 100% do tráfego",
    Recommendation.KEEP_CONTROL: "Manter experimento atual (sem alterações)",
    Recommendation.COLLECT_MORE_DATA: "Coletar mais dados antes de decidir",
    Recommendation.INCONCLUSIVE: "Inconclusivo — evidências insuficientes",
}


class ReportService:
    """Builds and persists experiment reports."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def build_report(
        self,
        experiment_name: str,
        partner: str,
        decision: Decision,
        narrative: str,
    ) -> ExperimentReport:
        """Assemble an ``ExperimentReport`` from pipeline outputs.

        Args:
            experiment_name: Human-readable experiment identifier.
            partner: Partner associated with the experiment.
            decision: The decision engine's output.
            narrative: LLM-generated executive narrative.

        Returns:
            A fully populated ``ExperimentReport``.
        """
        logger.info("Building report for experiment '%s'", experiment_name)

        stats = decision.statistical_summary.tests
        group_names = stats.get("group_names", [])
        n_variants = stats.get("n_groups", len(group_names))

        # Determine period from metrics (experiment_days).
        summary = decision.metrics_summary.summary
        days = max(
            (int(v.get("experiment_days", 0)) for v in summary.values()),
            default=0,
        )
        period = f"{days} dias"

        # Determine statistical significance.
        is_significant = self._is_significant(stats)

        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        return ExperimentReport(
            experiment_name=experiment_name,
            partner=partner,
            period=period,
            n_variants=int(n_variants),
            statistical_significance=is_significant,
            decision=decision,
            narrative=narrative,
            timestamp=timestamp,
        )

    def save(self, report: ExperimentReport) -> Path:
        """Persist *report* as Markdown and generate charts.

        Args:
            report: The report to save.

        Returns:
            Path to the saved Markdown report file.

        Raises:
            ReportGenerationError: If saving fails.
        """
        logger.info("Report save started for '%s'", report.experiment_name)

        try:
            output_dir = self._config.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            plots_dir = output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Generate charts.
            chart_paths = self._generate_charts(report, plots_dir)

            # Render Markdown.
            md_content = self._render_markdown(report, chart_paths)

            safe_name = report.experiment_name.replace(" ", "_").lower()
            md_path = output_dir / f"{safe_name}_report.md"
            md_path.write_text(md_content, encoding="utf-8")

            logger.info("Report saved to %s", md_path)
            return md_path

        except Exception as exc:
            logger.exception("Report save failed")
            raise ReportGenerationError("Failed to save report.") from exc

    # ── Markdown rendering ───────────────────────────────────────────

    def _render_markdown(
        self,
        report: ExperimentReport,
        chart_paths: list[Path],
    ) -> str:
        """Render the full Markdown report."""
        d = report.decision
        stats = d.statistical_summary.tests
        summary = d.metrics_summary.summary
        group_names = stats.get("group_names", [])

        sections = [
            self._section_title(report),
            self._section_executive_summary(report),
            self._section_experiment_overview(report),
            self._section_metrics_table(summary, group_names),
            self._section_statistical_findings(stats),
            self._section_recommendation(d),
            self._section_risks(d),
            self._section_charts(chart_paths),
        ]

        if report.narrative:
            sections.append(self._section_llm_narrative(report.narrative))

        return "\n\n".join(sections) + "\n"

    @staticmethod
    def _section_title(report: ExperimentReport) -> str:
        return f"# Relatório do Teste A/B — {report.experiment_name}"

    @staticmethod
    def _section_executive_summary(report: ExperimentReport) -> str:
        d = report.decision
        label = _RECOMMENDATION_LABELS.get(
            d.recommendation, d.recommendation.value,
        )
        lines = [
            "## Resumo Executivo",
            "",
            f"**Parceiro:** {report.partner}  ",
            f"**Período:** {report.period}  ",
            f"**Variantes:** {report.n_variants}  ",
            f"**Variante vencedora:** {d.winning_variant or 'N/A'}  ",
            f"**Recomendação:** {label}  ",
            f"**Confiança:** {d.confidence:.1%}  ",
        ]
        return "\n".join(lines)

    @staticmethod
    def _section_experiment_overview(report: ExperimentReport) -> str:
        d = report.decision
        lines = [
            "## Visão Geral do Experimento",
            "",
            d.justification,
        ]
        return "\n".join(lines)

    @staticmethod
    def _section_metrics_table(
        summary: dict[str, object],
        group_names: list[str],
    ) -> str:
        """Build a Markdown table of key metrics per group."""
        rows_spec = [
            ("Compradores Totais", "total_buyers", "{:,.0f}"),
            ("Vendas Totais", "total_sales", "R$ {:,.2f}"),
            ("Comissão Total", "total_commission", "R$ {:,.2f}"),
            ("Cashback Total", "total_cashback", "R$ {:,.2f}"),
            ("Receita Líquida Total", "total_net_revenue", "R$ {:,.2f}"),
            ("Média Diária de Compradores", "avg_daily_buyers", "{:,.1f}"),
            ("Média Diária de Vendas", "avg_daily_sales", "R$ {:,.2f}"),
            ("Média Diária de Receita Líquida", "avg_daily_net_revenue", "R$ {:,.2f}"),
            ("Vendas por Comprador", "sales_per_buyer", "R$ {:,.2f}"),
            ("Receita Líquida por Comprador", "net_revenue_per_buyer", "R$ {:,.2f}"),
            ("Taxa de Comissão", "commission_rate", "{:.2%}"),
            ("Taxa de Cashback", "cashback_rate", "{:.2%}"),
            ("Margem Líquida", "net_margin", "{:.2%}"),
            ("Dias de Experimento", "experiment_days", "{:.0f}"),
        ]

        header = "| Métrica | " + " | ".join(group_names) + " |"
        separator = "|---|" + "|".join(["---"] * len(group_names)) + "|"
        lines = ["## Métricas de Negócio", "", header, separator]

        for label, key, fmt in rows_spec:
            cells = []
            for g in group_names:
                val = summary.get(g, {}).get(key, 0)
                try:
                    cells.append(fmt.format(float(val)))
                except (ValueError, TypeError):
                    cells.append(str(val))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

        return "\n".join(lines)

    @staticmethod
    def _section_statistical_findings(stats: dict[str, object]) -> str:
        lines = ["## Descobertas Estatísticas", ""]
        n_groups = stats.get("n_groups", 0)
        target = stats.get("target_metric", "net_revenue")
        target_pt = "receita_liquida" if target == "net_revenue" else target
        lines.append(f"**Métrica alvo:** {target_pt}  ")
        lines.append(f"**Número de grupos:** {n_groups}  ")

        # Normality
        normality = stats.get("normality", {})
        if normality:
            lines.append("")
            lines.append("### Testes de Normalidade (Shapiro-Wilk)")
            lines.append("")
            for group, result in normality.items():
                p = result.get("p_value")
                normal = result.get("is_normal")
                note = result.get("note", "")
                if note:
                    lines.append(f"- **{group}:** {note}")
                else:
                    lines.append(
                        f"- **{group}:** p={p:.4f} — "
                        f"{'Normal' if normal else 'Não normal'}"
                    )

        # Primary test (2 groups)
        primary = stats.get("primary_test")
        if primary:
            lines.append("")
            lines.append("### Teste de Hipótese")
            lines.append("")
            lines.append(f"- **Método:** {primary.get('method')}")
            lines.append(f"- **Estatística:** {primary.get('statistic')}")
            lines.append(f"- **p-valor:** {primary.get('p_value')}")
            lines.append(
                f"- **Significativo:** "
                f"{'Sim' if primary.get('significant') else 'Não'}"
            )
            ci = primary.get("confidence_interval", {})
            if ci:
                lines.append(
                    f"- **IC de {ci.get('confidence_level', 0.95):.0%} para "
                    f"a diferença das médias:** [{ci.get('lower')}, {ci.get('upper')}]"
                )
            cd = primary.get("cohens_d", {})
            if cd:
                magnitude_pt = {
                    "large": "grande",
                    "medium": "médio",
                    "small": "pequeno",
                    "negligible": "desprezível",
                }.get(cd.get('magnitude'), cd.get('magnitude'))
                lines.append(
                    f"- **d de Cohen:** {cd.get('d')} ({magnitude_pt})"
                )

        # Omnibus test (3+ groups)
        omnibus = stats.get("omnibus_test")
        if omnibus:
            lines.append("")
            lines.append("### Teste Omnibus")
            lines.append("")
            lines.append(f"- **Método:** {omnibus.get('method')}")
            lines.append(f"- **Estatística:** {omnibus.get('statistic')}")
            lines.append(f"- **p-valor:** {omnibus.get('p_value')}")
            lines.append(
                f"- **Significativo:** "
                f"{'Sim' if omnibus.get('significant') else 'Não'}"
            )

        # Pairwise comparisons
        pairwise = stats.get("pairwise", [])
        if pairwise:
            lines.append("")
            lines.append("### Comparações em Pares (Bonferroni-corrigido)")
            lines.append("")
            for comp in pairwise:
                pair = f"{comp.get('group_a')} vs {comp.get('group_b')}"
                lines.append(f"**{pair}**")
                lines.append(f"- Método: {comp.get('method')}")
                lines.append(f"- p-valor: {comp.get('p_value')}")
                lines.append(
                    f"- Significativo (α={comp.get('adjusted_alpha')}): "
                    f"{'Sim' if comp.get('significant') else 'Não'}"
                )
                cd = comp.get("cohens_d", {})
                if cd:
                    magnitude_pt = {
                        "large": "grande",
                        "medium": "médio",
                        "small": "pequeno",
                        "negligible": "desprezível",
                    }.get(cd.get('magnitude'), cd.get('magnitude'))
                    lines.append(
                        f"- d de Cohen: {cd.get('d')} ({magnitude_pt})"
                    )
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _section_recommendation(d: Decision) -> str:
        label = _RECOMMENDATION_LABELS.get(
            d.recommendation, d.recommendation.value,
        )
        lines = [
            "## Recomendação Final",
            "",
            f"**Decisão:** {label}  ",
            f"**Variante vencedora:** {d.winning_variant or 'N/A'}  ",
            f"**Confiança:** {d.confidence:.1%}  ",
            "",
            d.justification,
        ]
        return "\n".join(lines)

    @staticmethod
    def _section_risks(d: Decision) -> str:
        lines = ["## Riscos", ""]
        if d.risks:
            for risk in d.risks:
                lines.append(f"- {risk}")
        else:
            lines.append("Nenhum risco significativo identificado.")
        return "\n".join(lines)

    @staticmethod
    def _section_charts(chart_paths: list[Path]) -> str:
        if not chart_paths:
            return ""
        lines = ["## Visualizações", ""]
        for p in chart_paths:
            lines.append(f"![{p.stem}](plots/{p.name})")
        return "\n".join(lines)

    @staticmethod
    def _section_llm_narrative(narrative: str) -> str:
        return f"## Análise Gerada por IA\n\n{narrative}"

    # ── Chart generation ─────────────────────────────────────────────

    def _generate_charts(
        self,
        report: ExperimentReport,
        plots_dir: Path,
    ) -> list[Path]:
        """Generate bar charts for key metrics and save as PNG."""
        summary = report.decision.metrics_summary.summary
        group_names = list(summary.keys())

        if not group_names:
            return []

        charts_spec = [
            ("total_net_revenue", "Receita Líquida por Variante", "R$"),
            ("total_sales", "Vendas Totais por Variante", "R$"),
            ("total_cashback", "Cashback Total por Variante", "R$"),
            ("total_buyers", "Compradores Totais por Variante", ""),
        ]

        paths: list[Path] = []
        safe_name = report.experiment_name.replace(" ", "_").lower()

        for metric_key, title, unit in charts_spec:
            values = [
                float(summary[g].get(metric_key, 0)) for g in group_names
            ]
            path = plots_dir / f"{safe_name}_{metric_key}.png"
            self._save_bar_chart(group_names, values, title, unit, path)
            paths.append(path)

        return paths

    @staticmethod
    def _save_bar_chart(
        labels: list[str],
        values: list[float],
        title: str,
        unit: str,
        path: Path,
    ) -> None:
        """Render and save a single bar chart."""
        colors = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"]
        bar_colors = colors[:len(labels)]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, values, color=bar_colors, edgecolor="white", width=0.5)

        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_ylabel(f"Valor ({unit})" if unit else "Valor", fontsize=11)
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Value labels on bars.
        for bar in bars:
            height = bar.get_height()
            label = f"{unit} {height:,.0f}" if unit else f"{height:,.0f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                label.strip(),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _is_significant(stats: dict[str, object]) -> bool:
        """Check if any test found statistical significance."""
        primary = stats.get("primary_test")
        if primary:
            return bool(primary.get("significant", False))

        omnibus = stats.get("omnibus_test")
        if omnibus:
            return bool(omnibus.get("significant", False))

        return False
