"""Decision engine service.

Combines business metrics and statistical results to produce an
actionable recommendation on which variant to scale.
"""

from __future__ import annotations

from src.exceptions import DecisionError
from src.logging_config import get_logger
from src.models import Decision, MetricsResult, Recommendation, StatisticalResult

logger = get_logger(__name__)

# Minimum absolute Cohen's d to consider an effect practically meaningful.
_MIN_EFFECT_SIZE = 0.2


class DecisionService:
    """Produces a business recommendation from analysis results."""

    def decide(
        self,
        metrics: MetricsResult,
        statistics: StatisticalResult,
    ) -> Decision:
        """Evaluate metrics and statistical evidence to form a decision.

        Args:
            metrics: Computed business metrics.
            statistics: Statistical test results.

        Returns:
            A ``Decision`` with recommendation and justification.

        Raises:
            DecisionError: If the decision cannot be produced.
        """
        logger.info("Decision engine started")

        try:
            n_groups = statistics.tests.get("n_groups", 0)

            if n_groups == 2:
                decision = self._decide_two_groups(metrics, statistics)
            elif n_groups > 2:
                decision = self._decide_multiple_groups(metrics, statistics)
            else:
                raise DecisionError(
                    f"Cannot decide with {n_groups} group(s)."
                )

            logger.info(
                "Decision engine completed — recommendation=%s",
                decision.recommendation.value,
            )
            return decision

        except DecisionError:
            raise
        except Exception as exc:
            logger.exception("Decision engine failed")
            raise DecisionError("Failed to produce a recommendation.") from exc

    # ── Two-group logic ──────────────────────────────────────────────

    def _decide_two_groups(
        self,
        metrics: MetricsResult,
        statistics: StatisticalResult,
    ) -> Decision:
        """Decision logic for exactly two experiment groups."""
        test = statistics.tests.get("primary_test", {})
        is_significant = bool(test.get("significant", False))
        p_value = float(test.get("p_value", 1.0))
        cohens = test.get("cohens_d", {})
        effect_d = float(cohens.get("d", 0.0))
        magnitude = str(cohens.get("magnitude", "negligible"))

        group_names = statistics.tests.get("group_names", [])
        best, runner_up = self._rank_groups_by_revenue(metrics, group_names)

        has_practical_effect = abs(effect_d) >= _MIN_EFFECT_SIZE
        best_has_higher_revenue = best == self._higher_revenue_group(
            effect_d, group_names,
        ) if len(group_names) == 2 else True

        return self._build_decision(
            is_significant=is_significant,
            has_practical_effect=has_practical_effect,
            best_variant=best,
            runner_up=runner_up,
            p_value=p_value,
            effect_d=effect_d,
            magnitude=magnitude,
            metrics=metrics,
            statistics=statistics,
        )

    # ── Multi-group logic ────────────────────────────────────────────

    def _decide_multiple_groups(
        self,
        metrics: MetricsResult,
        statistics: StatisticalResult,
    ) -> Decision:
        """Decision logic for three or more experiment groups."""
        omnibus = statistics.tests.get("omnibus_test", {})
        is_significant = bool(omnibus.get("significant", False))

        group_names = statistics.tests.get("group_names", [])
        best, runner_up = self._rank_groups_by_revenue(metrics, group_names)

        # Check pairwise results for the best group.
        pairwise = statistics.tests.get("pairwise", [])
        has_practical_effect = False
        best_p_value = 1.0
        best_effect_d = 0.0
        magnitude = "negligible"

        for comp in pairwise:
            involves_best = best in (comp.get("group_a"), comp.get("group_b"))
            if not involves_best:
                continue
            p = float(comp.get("p_value", 1.0))
            cd = comp.get("cohens_d", {})
            d = abs(float(cd.get("d", 0.0)))
            if p < best_p_value:
                best_p_value = p
                best_effect_d = d
                magnitude = str(cd.get("magnitude", "negligible"))
            if d >= _MIN_EFFECT_SIZE:
                has_practical_effect = True

        return self._build_decision(
            is_significant=is_significant and bool(pairwise),
            has_practical_effect=has_practical_effect,
            best_variant=best,
            runner_up=runner_up,
            p_value=best_p_value,
            effect_d=best_effect_d,
            magnitude=magnitude,
            metrics=metrics,
            statistics=statistics,
        )

    # ── Shared helpers ───────────────────────────────────────────────

    @staticmethod
    def _rank_groups_by_revenue(
        metrics: MetricsResult,
        group_names: list[str],
    ) -> tuple[str, str]:
        """Return (best, runner_up) by total net revenue."""
        summary = metrics.summary
        ranked = sorted(
            group_names,
            key=lambda g: float(summary.get(g, {}).get("total_net_revenue", 0)),
            reverse=True,
        )
        return ranked[0], ranked[1] if len(ranked) > 1 else ""

    @staticmethod
    def _higher_revenue_group(
        effect_d: float,
        group_names: list[str],
    ) -> str:
        """Determine which group the positive effect favors.

        When effect_d > 0, group_a (index 0) has the higher mean.
        """
        if len(group_names) < 2:
            return group_names[0] if group_names else ""
        return group_names[0] if effect_d >= 0 else group_names[1]

    @staticmethod
    def _build_decision(
        *,
        is_significant: bool,
        has_practical_effect: bool,
        best_variant: str,
        runner_up: str,
        p_value: float,
        effect_d: float,
        magnitude: str,
        metrics: MetricsResult,
        statistics: StatisticalResult,
    ) -> Decision:
        """Construct the Decision object from evaluation signals."""
        risks: list[str] = []

        magnitude_pt = {
            "large": "grande",
            "medium": "médio",
            "small": "pequeno",
            "negligible": "desprezível",
        }.get(magnitude, magnitude)

        # ── Strong evidence: significant + practical effect ──
        if is_significant and has_practical_effect:
            recommendation = Recommendation.SCALE_TREATMENT
            confidence = min(1.0 - p_value, 0.99)
            justification = (
                f"'{best_variant}' apresenta melhora estatisticamente significativa "
                f"(p={p_value:.4f}) com um efeito prático {magnitude_pt} "
                f"(d de Cohen={effect_d:.3f}). Recomenda-se escalar para todo o tráfego."
            )
            if magnitude == "small":
                risks.append(
                    "O tamanho do efeito é pequeno — os ganhos podem ser modestos em escala total."
                )

        # ── Significant but no practical effect ──
        elif is_significant and not has_practical_effect:
            recommendation = Recommendation.KEEP_CONTROL
            confidence = 0.5
            justification = (
                f"Significância estatística detectada (p={p_value:.4f}), mas o "
                f"efeito prático é {magnitude_pt} (d de Cohen={effect_d:.3f}). "
                f"É improvável que a diferença produza um impacto comercial significativo."
            )
            risks.append(
                "Resultados estatisticamente significativos sem significância "
                "prática podem não justificar o custo da mudança."
            )

        # ── Not significant but potentially meaningful effect ──
        elif not is_significant and has_practical_effect:
            recommendation = Recommendation.COLLECT_MORE_DATA
            confidence = 0.3
            justification = (
                f"'{best_variant}' mostra um efeito potencialmente significativo "
                f"(d de Cohen={effect_d:.3f}), mas o resultado não é "
                f"estatisticamente significativo (p={p_value:.4f}). "
                f"Mais dados podem confirmar ou refutar essa tendência."
            )
            risks.append(
                "A diferença observada pode ser devida à variação aleatória."
            )

        # ── Neither significant nor meaningful ──
        else:
            recommendation = Recommendation.INCONCLUSIVE
            confidence = 0.1
            justification = (
                f"Nenhuma diferença estatisticamente significativa detectada "
                f"(p={p_value:.4f}) e o efeito prático é {magnitude_pt}. "
                f"Não há evidências suficientes para recomendar qualquer variante."
            )
            risks.append(
                "A continuação do experimento sem alterações pode atrasar "
                "a tomada de decisão sem produzir novos insights."
            )

        return Decision(
            recommendation=recommendation,
            winning_variant=best_variant,
            confidence=round(confidence, 4),
            justification=justification,
            risks=risks,
            metrics_summary=metrics,
            statistical_summary=statistics,
        )
