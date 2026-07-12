"""Statistical analysis service.

Compares experiment variants using hypothesis tests to determine
whether observed differences are statistically significant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.config import StatisticsConfig
from src.exceptions import StatisticalAnalysisError
from src.logging_config import get_logger
from src.models import MetricsResult, StatisticalResult

logger = get_logger(__name__)

# Primary metric used for statistical comparison.
_TARGET_METRIC = "net_revenue"


class StatisticsService:
    """Runs statistical tests on experiment data."""

    def __init__(self, config: StatisticsConfig) -> None:
        self._config = config

    def analyze(
        self,
        data: pd.DataFrame,
        metrics: MetricsResult,
    ) -> StatisticalResult:
        """Perform statistical analysis comparing experiment variants.

        Automatically detects the number of groups and selects
        the appropriate test:
        - 2 groups → Welch's t-test or Mann-Whitney U
        - 3+ groups → ANOVA or Kruskal-Wallis (+ post-hoc if significant)

        Args:
            data: Preprocessed experiment DataFrame.
            metrics: Previously computed business metrics.

        Returns:
            A ``StatisticalResult`` with test outcomes.

        Raises:
            StatisticalAnalysisError: If the analysis cannot be completed.
        """
        alpha = self._config.significance_level
        logger.info("Statistical analysis started (α=%.2f)", alpha)

        try:
            groups = self._extract_groups(data, _TARGET_METRIC)
            group_names = list(groups.keys())
            n_groups = len(group_names)

            logger.info("Detected %d experiment group(s): %s", n_groups, group_names)

            normality = self._test_normality(groups)

            all_normal = all(r["is_normal"] for r in normality.values())
            logger.info("Normality assumption met: %s", all_normal)

            if n_groups == 2:
                tests = self._analyze_two_groups(
                    groups, group_names, normality, all_normal, alpha,
                )
            else:
                tests = self._analyze_multiple_groups(
                    groups, group_names, normality, all_normal, alpha,
                )

            tests["n_groups"] = n_groups
            tests["group_names"] = group_names
            tests["target_metric"] = _TARGET_METRIC
            tests["normality"] = normality

            logger.info("Statistical analysis completed")
            return StatisticalResult(tests=tests)

        except StatisticalAnalysisError:
            raise
        except Exception as exc:
            logger.exception("Statistical analysis failed")
            raise StatisticalAnalysisError(
                "Unexpected error during statistical analysis."
            ) from exc

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_groups(
        data: pd.DataFrame,
        metric: str,
    ) -> dict[str, np.ndarray]:
        """Extract per-group arrays for the given metric."""
        groups: dict[str, np.ndarray] = {}
        for name, frame in data.groupby("user_group"):
            values = frame[metric].dropna().to_numpy(dtype=float)
            groups[str(name)] = values
        return groups

    @staticmethod
    def _test_normality(
        groups: dict[str, np.ndarray],
    ) -> dict[str, dict[str, object]]:
        """Run Shapiro-Wilk normality test on each group."""
        results: dict[str, dict[str, object]] = {}
        for name, values in groups.items():
            if len(values) < 8:
                results[name] = {
                    "statistic": None,
                    "p_value": None,
                    "is_normal": False,
                    "note": "Amostra muito pequena para o teste de normalidade",
                }
                continue

            if np.std(values) == 0:
                results[name] = {
                    "statistic": None,
                    "p_value": None,
                    "is_normal": False,
                    "note": "Variância zero — todos os valores são idênticos",
                }
                continue

            stat, p = sp_stats.shapiro(values)
            results[name] = {
                "statistic": round(float(stat), 6),
                "p_value": round(float(p), 6),
                "is_normal": bool(p > 0.05),
            }
        return results

    # ── Two-group analysis ───────────────────────────────────────────

    def _analyze_two_groups(
        self,
        groups: dict[str, np.ndarray],
        names: list[str],
        normality: dict[str, dict[str, object]],
        all_normal: bool,
        alpha: float,
    ) -> dict[str, object]:
        """Compare exactly two groups."""
        a, b = groups[names[0]], groups[names[1]]

        if all_normal:
            test_result = self._welch_t_test(a, b, alpha)
        else:
            test_result = self._mann_whitney(a, b, alpha)

        test_result["confidence_interval"] = self._mean_diff_ci(
            a, b, self._config.confidence_level,
        )
        test_result["cohens_d"] = self._cohens_d(a, b)

        return {"primary_test": test_result}

    @staticmethod
    def _welch_t_test(
        a: np.ndarray, b: np.ndarray, alpha: float,
    ) -> dict[str, object]:
        """Welch's t-test (unequal variances)."""
        stat, p = sp_stats.ttest_ind(a, b, equal_var=False)
        return {
            "method": "Teste t de Welch",
            "statistic": round(float(stat), 6),
            "p_value": round(float(p), 6),
            "significant": bool(p < alpha),
        }

    @staticmethod
    def _mann_whitney(
        a: np.ndarray, b: np.ndarray, alpha: float,
    ) -> dict[str, object]:
        """Mann-Whitney U test (non-parametric alternative)."""
        stat, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
        return {
            "method": "U de Mann-Whitney",
            "statistic": round(float(stat), 6),
            "p_value": round(float(p), 6),
            "significant": bool(p < alpha),
        }

    @staticmethod
    def _mean_diff_ci(
        a: np.ndarray,
        b: np.ndarray,
        confidence: float,
    ) -> dict[str, float]:
        """Confidence interval for the difference in means."""
        diff = float(np.mean(a) - np.mean(b))
        se = float(np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)))
        df = len(a) + len(b) - 2
        t_crit = float(sp_stats.t.ppf((1 + confidence) / 2, df))
        return {
            "mean_difference": round(diff, 2),
            "lower": round(diff - t_crit * se, 2),
            "upper": round(diff + t_crit * se, 2),
            "confidence_level": confidence,
        }

    @staticmethod
    def _cohens_d(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
        """Compute Cohen's d effect size."""
        n_a, n_b = len(a), len(b)
        pooled_std = float(np.sqrt(
            ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1))
            / (n_a + n_b - 2)
        ))
        if pooled_std == 0:
            d = 0.0
        else:
            d = float((np.mean(a) - np.mean(b)) / pooled_std)

        magnitude = "negligible"
        abs_d = abs(d)
        if abs_d >= 0.8:
            magnitude = "large"
        elif abs_d >= 0.5:
            magnitude = "medium"
        elif abs_d >= 0.2:
            magnitude = "small"

        return {"d": round(d, 4), "magnitude": magnitude}

    # ── Multi-group analysis ─────────────────────────────────────────

    def _analyze_multiple_groups(
        self,
        groups: dict[str, np.ndarray],
        names: list[str],
        normality: dict[str, dict[str, object]],
        all_normal: bool,
        alpha: float,
    ) -> dict[str, object]:
        """Compare three or more groups."""
        arrays = [groups[n] for n in names]

        if all_normal:
            omnibus = self._anova(arrays, alpha)
        else:
            omnibus = self._kruskal_wallis(arrays, alpha)

        result: dict[str, object] = {"omnibus_test": omnibus}

        if omnibus["significant"]:
            result["pairwise"] = self._pairwise_comparisons(
                groups, names, all_normal, alpha,
            )

        return result

    @staticmethod
    def _anova(arrays: list[np.ndarray], alpha: float) -> dict[str, object]:
        """One-way ANOVA."""
        stat, p = sp_stats.f_oneway(*arrays)
        return {
            "method": "ANOVA de uma via",
            "statistic": round(float(stat), 6),
            "p_value": round(float(p), 6),
            "significant": bool(p < alpha),
        }

    @staticmethod
    def _kruskal_wallis(
        arrays: list[np.ndarray], alpha: float,
    ) -> dict[str, object]:
        """Kruskal-Wallis H-test (non-parametric alternative to ANOVA)."""
        stat, p = sp_stats.kruskal(*arrays)
        return {
            "method": "Kruskal-Wallis",
            "statistic": round(float(stat), 6),
            "p_value": round(float(p), 6),
            "significant": bool(p < alpha),
        }

    def _pairwise_comparisons(
        self,
        groups: dict[str, np.ndarray],
        names: list[str],
        all_normal: bool,
        alpha: float,
    ) -> list[dict[str, object]]:
        """Post-hoc pairwise comparisons with Bonferroni correction."""
        n_comparisons = len(names) * (len(names) - 1) // 2
        adjusted_alpha = alpha / n_comparisons

        comparisons: list[dict[str, object]] = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = groups[names[i]], groups[names[j]]

                if all_normal:
                    test = self._welch_t_test(a, b, adjusted_alpha)
                else:
                    test = self._mann_whitney(a, b, adjusted_alpha)

                test["group_a"] = names[i]
                test["group_b"] = names[j]
                test["adjusted_alpha"] = round(adjusted_alpha, 4)
                test["cohens_d"] = self._cohens_d(a, b)
                test["confidence_interval"] = self._mean_diff_ci(
                    a, b, self._config.confidence_level,
                )
                comparisons.append(test)

        return comparisons
