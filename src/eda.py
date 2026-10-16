"""
eda.py
------
Generates all EDA figures with reasoning embedded in function docstrings.

Every chart answers a specific business or data quality question.
The question is stated in the docstring so the notebook narrative
can reference it directly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def _setup() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", palette="Set2")


def plot_missing_values(df: pd.DataFrame) -> None:
    """
    Question: Which columns have missing data, and how much?

    Why: Missing data is not random — its pattern tells us about the
    data collection process. Customer ID missing = guest checkout (cannot
    profile). Description missing = potential data entry error.
    We visualize this before any cleaning so we can justify every removal.
    """
    _setup()
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        print("  No missing values found.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    missing.plot(kind="bar", ax=ax, color="salmon")
    ax.set(title="Missing Values (% of total rows)",
           xlabel="Column", ylabel="Missing %")
    ax.axhline(5, linestyle="--", color="red", alpha=0.5, label="5% threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_01_missing_values.png", dpi=150)
    plt.close(fig)


def plot_revenue_over_time(df_clean: pd.DataFrame) -> None:
    """
    Question: Is the business growing, declining, or seasonal?

    Why: Revenue trend tells us if the dataset is from a healthy period
    or a declining one — which affects how we interpret 'Recency'.
    A declining business has more naturally high-Recency customers
    (they stopped because the business stopped serving them, not by choice).
    """
    _setup()
    df = df_clean.copy()
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
    monthly = df.groupby("YearMonth")["Revenue"].sum().reset_index()
    monthly["YearMonth"] = monthly["YearMonth"].astype(str)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly["YearMonth"], monthly["Revenue"], marker="o", linewidth=2)
    ax.fill_between(range(len(monthly)), monthly["Revenue"], alpha=0.15)
    ax.set_xticklabels(monthly["YearMonth"], rotation=45, ha="right")
    ax.set(title="Monthly Revenue Trend — Is the Business Growing?",
           xlabel="Month", ylabel="Total Revenue (£)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_02_revenue_trend.png", dpi=150)
    plt.close(fig)


def plot_rfm_distributions(customer_features: pd.DataFrame) -> None:
    """
    Question: Are RFM values skewed? Are there extreme outliers?

    Why: K-Means and GMM use Euclidean distance — they break down when
    one customer's Monetary is 1,000x the average. Seeing the distribution
    BEFORE modeling confirms we need log transformation in preprocessing.
    A right-skewed Monetary distribution is expected and justifies log scaling.
    """
    _setup()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, color in zip(axes, ["Recency", "Frequency", "Monetary"],
                               ["steelblue", "seagreen", "coral"]):
        sns.histplot(customer_features[col], bins=40, kde=True, ax=ax, color=color)
        ax.set_title(f"{col} Distribution\n(skewed → needs log transform before clustering)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_03_rfm_distributions.png", dpi=150)
    plt.close(fig)


def plot_top_countries(df_clean: pd.DataFrame, top_n: int = 10) -> None:
    """
    Question: Which countries drive the most revenue? Is this mostly a UK business?

    Why: If 95%+ of revenue is from one country, country is not a useful
    segmentation feature and should be dropped. This confirms our decision
    to exclude Country from modeling.
    """
    _setup()
    by_country = (
        df_clean.groupby("Country")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=by_country.values, y=by_country.index, ax=ax)
    ax.set(title=f"Top {top_n} Countries by Revenue",
           xlabel="Total Revenue (£)", ylabel="Country")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_04_top_countries.png", dpi=150)
    plt.close(fig)


def plot_purchase_hour_heatmap(df_clean: pd.DataFrame) -> None:
    """
    Question: When do customers shop? Is there a weekly pattern?

    Why: Time patterns reveal customer behavior that pure spending metrics
    miss. A customer who always shops at 2am may be an international buyer
    or a night-shift worker — this informs personalization strategy.
    This also validates our 'PreferredHour' feature.
    """
    _setup()
    df = df_clean.copy()
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    pivot = df.pivot_table(
        values="Revenue", index="DayOfWeek", columns="Hour",
        aggfunc="sum", fill_value=0,
    )
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3)
    ax.set(title="Revenue Heatmap by Day and Hour — When Do Customers Shop?",
           xlabel="Hour of Day", ylabel="Day of Week")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_05_purchase_heatmap.png", dpi=150)
    plt.close(fig)


def plot_customer_value_distribution(customer_features: pd.DataFrame) -> None:
    """
    Question: What percentage of customers drive 80% of revenue? (Pareto check)

    Why: If 20% of customers drive 80% of revenue, we need a segmentation
    strategy that protects and grows that top 20% — not a one-size-fits-all
    campaign. This justifies the 'IsHighValue' classification label.
    """
    _setup()
    sorted_monetary = customer_features["Monetary"].sort_values(ascending=False)
    cumulative_pct = sorted_monetary.cumsum() / sorted_monetary.sum() * 100
    customer_pct = np.arange(1, len(sorted_monetary) + 1) / len(sorted_monetary) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(customer_pct, cumulative_pct, linewidth=2)
    ax.axhline(80, linestyle="--", color="red", alpha=0.6, label="80% revenue")
    ax.axvline(20, linestyle="--", color="blue", alpha=0.6, label="Top 20% customers")
    ax.set(title="Pareto Curve — Do Top 20% Customers Drive 80% Revenue?",
           xlabel="Cumulative % of Customers (sorted by spend)",
           ylabel="Cumulative % of Total Revenue")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_06_pareto_curve.png", dpi=150)
    plt.close(fig)


def run_full_eda(df_raw: pd.DataFrame, df_clean: pd.DataFrame,
                 customer_features: pd.DataFrame) -> None:
    """Run all EDA plots in sequence."""
    print("Generating EDA figures...")
    plot_missing_values(df_raw)
    plot_revenue_over_time(df_clean)
    plot_rfm_distributions(customer_features)
    plot_top_countries(df_clean)
    plot_purchase_hour_heatmap(df_clean)
    plot_customer_value_distribution(customer_features)
    print(f"  All EDA figures saved to {FIGURES_DIR}")
