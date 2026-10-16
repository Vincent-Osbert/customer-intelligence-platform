"""
preprocessing.py
----------------
Cleans the raw Online Retail II data and prepares it for feature engineering.

Every decision here is intentional. Comments explain WHY each step is done,
not just WHAT it does.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert InvoiceDate from string to datetime.

    Why: Downstream features like Recency and purchase-hour patterns
    depend on proper datetime objects. Keeping it as a string would
    silently fail during groupby operations.
    """
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df


def remove_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where Invoice starts with 'C' (cancelled/return transactions).

    Why: Return transactions have negative Quantity. Including them inflates
    Recency but deflates Monetary and Frequency — distorting the customer
    profile. We keep a separate audit count but exclude them from modeling.
    Returns are a separate analytical problem (return-rate analysis).
    """
    n_before = len(df)
    df = df[~df["Invoice"].astype(str).str.startswith("C")].copy()
    n_removed = n_before - len(df)
    print(f"  Removed {n_removed:,} return/cancellation rows.")
    return df


def remove_invalid_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where Quantity > 0 and Price > 0.

    Why: Zero-price rows are typically internal adjustments or test entries
    by the retailer's system. Negative quantities that are NOT cancellations
    are data errors. Both types would create misleading revenue figures.
    """
    n_before = len(df)
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()
    print(f"  Removed {n_before - len(df):,} rows with Quantity<=0 or Price<=0.")
    return df


def drop_missing_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with no Customer ID.

    Why: Without a Customer ID we cannot build any customer-level profile.
    These rows likely belong to walk-in or guest transactions — interesting
    for sales analysis but useless for customer segmentation.
    Why NOT impute: Inventing a customer ID would fabricate behavioral patterns.
    """
    n_before = len(df)
    df = df.dropna(subset=["Customer ID"]).copy()
    df["Customer ID"] = df["Customer ID"].astype(int)
    print(f"  Dropped {n_before - len(df):,} rows with missing Customer ID.")
    return df


def add_revenue_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a Revenue column (Quantity × Price) for each line item.

    Why: Revenue is the most direct business metric. Computing it once here
    avoids repeated multiplication across multiple analysis steps.
    """
    df = df.copy()
    df["Revenue"] = df["Quantity"] * df["Price"]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all cleaning steps in the correct order and return the cleaned DataFrame.
    """
    print("Running data cleaning pipeline...")
    df = parse_dates(df)
    df = remove_returns(df)
    df = remove_invalid_quantities(df)
    df = drop_missing_customers(df)
    df = add_revenue_column(df)
    print(f"  Clean data: {len(df):,} rows remaining.")
    return df


def save_clean(df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "clean_transactions.csv"
    df.to_csv(out, index=False)
    print(f"  Clean data saved → {out}")
    return out
