"""
simulate_live.py
----------------
Simulates new customer transactions arriving daily.

In a real production system, this module would be replaced by a connection
to a live database or event stream (Kafka, Pub/Sub, etc.).

For this portfolio project, it appends realistic synthetic transactions
to a 'live feed' CSV file that the Streamlit dashboard reads from.
Running this script simulates "a new day of business" — the dashboard
will reflect the new data on its next auto-refresh cycle.

Usage:
    python simulate_live.py            # adds 1 day of new transactions
    python simulate_live.py --days 7   # adds 7 days at once
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
LIVE_FILE = PROCESSED_DIR / "live_transactions.csv"
RANDOM_STATE = 42

# Load existing customer pool from clean data if available
CLEAN_FILE = PROCESSED_DIR / "clean_transactions.csv"


def _get_customer_pool() -> list[int]:
    """Return a list of existing customer IDs to simulate realistic returning buyers."""
    if CLEAN_FILE.exists():
        df = pd.read_csv(CLEAN_FILE, usecols=["Customer ID"], nrows=5000)
        return df["Customer ID"].dropna().unique().tolist()
    return list(range(10000, 10500))   # fallback synthetic pool


def generate_daily_transactions(
    transaction_date: datetime,
    n_transactions: int = 120,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate a realistic day of transactions.

    Transaction count peaks mid-week (Tuesday–Thursday) and dips on weekends,
    mirroring the pattern we observed in the EDA heatmap.
    """
    rng = np.random.default_rng(seed or int(transaction_date.timestamp()))
    customer_pool = _get_customer_pool()

    # Weekday volume adjustment
    weekday = transaction_date.weekday()
    volume_multiplier = {0: 0.9, 1: 1.1, 2: 1.15, 3: 1.1, 4: 1.0, 5: 0.4, 6: 0.2}
    n = max(10, int(n_transactions * volume_multiplier.get(weekday, 1.0)))

    customers = rng.choice(customer_pool, size=n)
    quantities = rng.integers(1, 15, size=n)
    prices = np.round(rng.exponential(scale=15, size=n) + 2, 2)
    hours = rng.choice(range(8, 20), size=n, p=[
        0.03, 0.06, 0.10, 0.13, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.04, 0.01
    ])

    timestamps = [
        transaction_date.replace(hour=int(h), minute=random.randint(0, 59))
        for h in hours
    ]

    df = pd.DataFrame({
        "Invoice": [f"LIVE-{int(transaction_date.timestamp())}-{i}" for i in range(n)],
        "Customer ID": customers.astype(int),
        "InvoiceDate": timestamps,
        "Quantity": quantities,
        "Price": prices,
        "Revenue": np.round(quantities * prices, 2),
        "Country": rng.choice(["United Kingdom", "Germany", "France", "Netherlands"],
                               size=n, p=[0.85, 0.07, 0.05, 0.03]),
    })
    return df


def append_live_data(days: int = 1) -> None:
    """Append N days of simulated transactions to the live feed file."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if LIVE_FILE.exists():
        existing = pd.read_csv(LIVE_FILE, parse_dates=["InvoiceDate"])
        last_date = pd.to_datetime(existing["InvoiceDate"]).max()
        start_date = last_date + timedelta(days=1)
    else:
        existing = pd.DataFrame()
        start_date = datetime(2025, 1, 1)

    new_batches = []
    for i in range(days):
        day = start_date + timedelta(days=i)
        batch = generate_daily_transactions(day)
        new_batches.append(batch)
        print(f"  Simulated {len(batch)} transactions for {day.strftime('%Y-%m-%d')}")

    new_data = pd.concat(new_batches, ignore_index=True)
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined.to_csv(LIVE_FILE, index=False)
    print(f"  Live feed updated: {len(combined):,} total rows → {LIVE_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate new daily transactions.")
    parser.add_argument("--days", type=int, default=1, help="Number of days to simulate.")
    args = parser.parse_args()
    append_live_data(days=args.days)
