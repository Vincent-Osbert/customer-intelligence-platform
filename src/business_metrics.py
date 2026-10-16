"""
business_metrics.py
--------------------
Computes business-level outputs from segment data:
  - Customer Lifetime Value (CLV) estimate per segment
  - Action recommendations per segment
  - Segment size and revenue share

These outputs are designed to be read directly by a business/marketing team
with no data science background.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

# Assumptions (clearly stated — replace with real company data for production)
AVG_GROSS_MARGIN = 0.35      # 35% gross margin on each purchase
MONTHLY_CHURN_RATE = 0.05    # average 5% of customers leave per month
DISCOUNT_RATE = 0.10         # 10% annual discount rate for NPV calculation


def estimate_clv(avg_monthly_revenue: float, months: int = 12) -> float:
    """
    Simple CLV estimate using average revenue × retention.

    Formula: CLV = Margin × (Retention / (1 + Discount - Retention))
    This is a simplified version of the Pareto/NBD model.
    For a portfolio project it demonstrates understanding of CLV logic;
    a production system would use BG/NBD or probabilistic models.
    """
    retention = 1 - MONTHLY_CHURN_RATE
    monthly_discount = DISCOUNT_RATE / 12
    clv = (avg_monthly_revenue * AVG_GROSS_MARGIN) * (retention / (1 + monthly_discount - retention))
    return round(clv, 2)


# Recommended actions per segment — based on marketing best practices
SEGMENT_ACTIONS = {
    "VIP Champions": {
        "priority": "HIGH — Protect",
        "action": (
            "Reward loyalty: early access to new products, exclusive discounts, "
            "VIP customer events. Do NOT send generic mass promotions — "
            "they find it annoying and it cheapens the relationship."
        ),
    },
    "High Value at Risk": {
        "priority": "URGENT — Win Back",
        "action": (
            "Personal outreach within 7 days. Offer a meaningful incentive "
            "(e.g., 20% off next order, free shipping). Ask for feedback — "
            "they likely stopped for a reason (bad experience, competitor offer)."
        ),
    },
    "Regular Engaged": {
        "priority": "MEDIUM — Grow",
        "action": (
            "Upsell strategy: recommend higher-margin products based on purchase history. "
            "Cross-sell bundles. Target with mid-tier loyalty program to move them "
            "toward VIP status."
        ),
    },
    "Occasional Buyers": {
        "priority": "LOW — Nurture",
        "action": (
            "Low-cost email nurture campaign. Seasonal promotions. "
            "Do not over-invest — ROI is uncertain for this group."
        ),
    },
    "Dormant / Lost": {
        "priority": "VERY LOW — Evaluate",
        "action": (
            "Minimal spend. One re-engagement email maximum. "
            "If no response, remove from active campaign lists to reduce costs. "
            "Analyze WHY they left to prevent future churn from other segments."
        ),
    },
}


def build_segment_report(
    segments_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a business-readable report combining segment profiles,
    CLV estimates, and recommended actions.

    segments_df: output of clustering.py (has Customer ID + Segment column)
    features_df: output of features.py (has Recency, Frequency, Monetary per customer)
    """
    merged = segments_df.merge(
        features_df[["Customer ID", "Recency", "Frequency", "Monetary"]],
        on="Customer ID",
        how="left",
    )

    report = (
        merged.groupby("Segment")
        .agg(
            CustomerCount=("Customer ID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonthlyRevenue=("Monetary", lambda x: x.mean() / 12),
            TotalRevenue=("Monetary", "sum"),
        )
        .reset_index()
    )

    total_revenue = report["TotalRevenue"].sum()
    report["RevenueShare_pct"] = (report["TotalRevenue"] / total_revenue * 100).round(1)
    report["EstimatedCLV"] = report["AvgMonthlyRevenue"].apply(estimate_clv)

    # Add action recommendations
    report["Priority"] = report["Segment"].map(
        lambda s: SEGMENT_ACTIONS.get(s, {}).get("priority", "—")
    )
    report["RecommendedAction"] = report["Segment"].map(
        lambda s: SEGMENT_ACTIONS.get(s, {}).get("action", "—")
    )

    report = report.round({"AvgRecency": 1, "AvgFrequency": 1,
                           "AvgMonthlyRevenue": 2, "TotalRevenue": 2})

    out = REPORTS_DIR / "segment_business_report.csv"
    report.to_csv(out, index=False)
    print(f"  Business report saved → {out}")
    return report
