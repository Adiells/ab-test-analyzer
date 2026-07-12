"""Business metrics computation service.

Calculates key performance indicators for each experiment variant.
This module only computes values — it does not interpret them.
"""

from __future__ import annotations

import pandas as pd

from src.exceptions import MetricsComputationError
from src.logging_config import get_logger
from src.models import MetricsResult

logger = get_logger(__name__)


class MetricsService:
    """Computes descriptive business metrics for each experiment variant."""

    def compute(self, data: pd.DataFrame) -> MetricsResult:
        """Compute business metrics from a preprocessed dataset.

        Args:
            data: Preprocessed experiment DataFrame.

        Returns:
            MetricsResult containing aggregated metrics for each variant.

        Raises:
            MetricsComputationError: If metric computation fails.
        """
        logger.info("Metrics computation started")

        try:
            grouped = data.groupby("user_group")

            aggregated = grouped.agg(
                total_buyers=("buyers", "sum"),
                total_sales=("total_sales", "sum"),
                total_commission=("commission", "sum"),
                total_cashback=("cashback", "sum"),
                total_net_revenue=("net_revenue", "sum"),
                avg_daily_buyers=("buyers", "mean"),
                avg_daily_sales=("total_sales", "mean"),
                avg_daily_commission=("commission", "mean"),
                avg_daily_cashback=("cashback", "mean"),
                avg_daily_net_revenue=("net_revenue", "mean"),
                std_buyers=("buyers", "std"),
                std_sales=("total_sales", "std"),
                std_commission=("commission", "std"),
                std_cashback=("cashback", "std"),
                std_net_revenue=("net_revenue", "std"),
                experiment_days=("buyers", "count"),
            )

            # Per-buyer metrics
            aggregated["sales_per_buyer"] = (
                aggregated["total_sales"] / aggregated["total_buyers"]
            ).round(2)

            aggregated["commission_per_buyer"] = (
                aggregated["total_commission"] / aggregated["total_buyers"]
            ).round(2)

            aggregated["cashback_per_buyer"] = (
                aggregated["total_cashback"] / aggregated["total_buyers"]
            ).round(2)

            aggregated["net_revenue_per_buyer"] = (
                aggregated["total_net_revenue"] / aggregated["total_buyers"]
            ).round(2)

            # Financial ratios
            aggregated["commission_rate"] = (
                aggregated["total_commission"] / aggregated["total_sales"]
            ).round(4)

            aggregated["cashback_rate"] = (
                aggregated["total_cashback"] / aggregated["total_sales"]
            ).round(4)

            aggregated["net_margin"] = (
                aggregated["total_net_revenue"] / aggregated["total_sales"]
            ).round(4)

            logger.info(
                "Metrics computation completed — %d group(s)",
                len(aggregated),
            )

            return MetricsResult(
                summary=aggregated.to_dict(orient="index")
            )

        except Exception as exc:
            logger.exception("Metric computation failed")
            raise MetricsComputationError(
                "Failed to compute business metrics."
            ) from exc