"""
features.py
-----------
Builds customer-level features from cleaned transaction data.

Input:  cleaned transaction DataFrame (one row per line item)
Output: one row per customer with RFM + behavioral features

Feature groups:
  1. RFM        — classic recency/frequency/monetary
  2. Basket     — how the customer shops per visit
  3. Temporal   — when the customer tends to shop
  4. Diversity  — breadth of products purchased
  5. Returns    — estimated return behavior (from pre-clean data)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary per customer.

    snapshot_date: the reference "today". Defaults to max(InvoiceDate) + 1 day.

    Why +1 day?  If a customer bought on the last day in the dataset their
    Recency would be 0.  Log(0) is undefined, so we shift by one day to
    guarantee Recency >= 1 for all customers.
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
    ).reset_index()
    return rfm


def compute_basket_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-customer basket-level statistics.

    AvgBasketSize:  average number of items per invoice — high = bulk buyer.
    AvgOrderValue:  average revenue per invoice — high = premium buyer.
    MaxOrderValue:  largest single order — captures occasional big spenders.
    """
    basket = df.groupby(["Customer ID", "Invoice"]).agg(
        basket_size=("Quantity", "sum"),
        order_value=("Revenue", "sum"),
    ).reset_index()

    features = basket.groupby("Customer ID").agg(
        AvgBasketSize=("basket_size", "mean"),
        AvgOrderValue=("order_value", "mean"),
        MaxOrderValue=("order_value", "max"),
    ).reset_index()
    return features


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-customer time-of-purchase patterns.

    PurchaseDaySpread: days between first and last purchase.
                       Short spread + high frequency = loyal repeat buyer.
    AvgDaysBetweenOrders: how often they come back on average.
    PreferredHour: modal hour of purchase (0-23) — useful for targeting emails.
    PreferredDayOfWeek: modal weekday (0=Mon, 6=Sun).
    """
    df = df.copy()
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek

    first_last = df.groupby("Customer ID")["InvoiceDate"].agg(["min", "max"])
    first_last["PurchaseDaySpread"] = (first_last["max"] - first_last["min"]).dt.days

    freq = df.groupby("Customer ID")["Invoice"].nunique().rename("n_orders")
    spread_freq = first_last.join(freq)
    spread_freq["AvgDaysBetweenOrders"] = (
        spread_freq["PurchaseDaySpread"] / spread_freq["n_orders"].clip(lower=1)
    )

    hour_mode = df.groupby("Customer ID")["Hour"].agg(lambda x: x.mode().iloc[0])
    dow_mode = df.groupby("Customer ID")["DayOfWeek"].agg(lambda x: x.mode().iloc[0])

    temporal = pd.DataFrame({
        "PurchaseDaySpread": spread_freq["PurchaseDaySpread"],
        "AvgDaysBetweenOrders": spread_freq["AvgDaysBetweenOrders"],
        "PreferredHour": hour_mode,
        "PreferredDayOfWeek": dow_mode,
    }).reset_index()
    return temporal


def compute_diversity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    How broadly a customer shops across product categories.

    UniqueProducts: number of distinct StockCodes purchased.
    UniqueCountries: should be 1 for most — flags multi-country accounts if >1.

    Why diversity matters: A customer who buys 50 different products is
    fundamentally different from one who always reorders the same item,
    even if their RFM scores are identical.
    """
    diversity = df.groupby("Customer ID").agg(
        UniqueProducts=("StockCode", "nunique"),
    ).reset_index()
    return diversity


def build_customer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all feature groups into a single customer-level DataFrame.

    Merging is done with left joins on Customer ID to guarantee no customers
    are lost in case a feature group returns fewer rows (e.g., customers with
    no spread because they only bought once).
    """
    print("Building customer-level features...")
    rfm = compute_rfm(df_clean)
    basket = compute_basket_features(df_clean)
    temporal = compute_temporal_features(df_clean)
    diversity = compute_diversity_features(df_clean)

    features = (
        rfm
        .merge(basket, on="Customer ID", how="left")
        .merge(temporal, on="Customer ID", how="left")
        .merge(diversity, on="Customer ID", how="left")
    )
    features = features.fillna(0)
    print(f"  Feature matrix shape: {features.shape}")
    return features


def add_classification_labels(features: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary labels for supervised classification.

    These labels are derived from the feature data itself — they represent
    business-relevant behavioral thresholds, not ground truth from a survey.

    IsHighValue:  Monetary above the 75th percentile.
                  Why 75th?  Top 25% captures the meaningful high-value group
                  without being so strict (90th+) that the class becomes tiny.

    IsFrequent:   Frequency above the median.
                  Why median?  Splits the customer base evenly into
                  engaged vs casual — a natural business divide.

    IsChurning:   Recency > 90 days AND customer has bought more than once.
                  Why 90 days?  For a retail store, 3 months without a purchase
                  is a common industry threshold for "at-risk" status.
                  The Frequency > 1 filter excludes one-time buyers who were
                  never truly "retained" in the first place.
    """
    df = features.copy()
    monetary_75th = df["Monetary"].quantile(0.75)
    frequency_median = df["Frequency"].median()

    df["IsHighValue"] = (df["Monetary"] >= monetary_75th).astype(int)
    df["IsFrequent"] = (df["Frequency"] >= frequency_median).astype(int)
    df["IsChurning"] = (
        (df["Recency"] > 90) & (df["Frequency"] > 1)
    ).astype(int)

    print(f"  Label rates — HighValue: {df['IsHighValue'].mean():.1%} | "
          f"Frequent: {df['IsFrequent'].mean():.1%} | "
          f"Churning: {df['IsChurning'].mean():.1%}")
    return df


def save_features(df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "customer_features.csv"
    df.to_csv(out, index=False)
    print(f"  Features saved → {out}")
    return out
