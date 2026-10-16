"""
train.py
--------
Entry point for the full Customer Intelligence Platform pipeline.

Run this script once to:
  1. Pull data from UCI ML Repository
  2. Clean and validate the data
  3. Engineer customer-level features
  4. Run EDA and save all figures
  5. Train classification models (IsHighValue, IsFrequent, IsChurning)
  6. Run clustering on predicted probability scores
  7. Compute business metrics and segment report
  8. Simulate initial live data feed for the dashboard

After this script completes, launch the dashboard with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion import load_raw, validate_schema, get_data_summary
from src.preprocessing import clean, save_clean
from src.features import build_customer_features, add_classification_labels, save_features
from src.eda import run_full_eda
from src.classification import run_classification
from src.clustering import run_clustering
from src.business_metrics import build_segment_report
from src.simulate_live import append_live_data


def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("  Customer Intelligence Platform — Training Pipeline")
    print("=" * 60)

    # ── Step 1: Data Ingestion ─────────────────────────────────
    print("\n[1/7] Data Ingestion")
    df_raw = load_raw()
    validate_schema(df_raw)
    summary = get_data_summary(df_raw)
    print(f"  Customers: {summary['unique_customers']:,} | "
          f"Products: {summary['unique_products']:,}")

    # ── Step 2: Preprocessing ──────────────────────────────────
    print("\n[2/7] Preprocessing")
    df_clean = clean(df_raw)
    save_clean(df_clean)

    # ── Step 3: Feature Engineering ───────────────────────────
    print("\n[3/7] Feature Engineering")
    features = build_customer_features(df_clean)
    features = add_classification_labels(features)
    save_features(features)

    # ── Step 4: EDA ───────────────────────────────────────────
    print("\n[4/7] Exploratory Data Analysis")
    run_full_eda(df_raw, df_clean, features)

    # ── Step 5: Classification ────────────────────────────────
    print("\n[5/7] Classification Models")
    scores_df, clf_metrics = run_classification(features)

    # ── Step 6: Clustering ────────────────────────────────────
    print("\n[6/7] Clustering")
    segments_df = run_clustering(scores_df)

    # ── Step 7: Business Metrics ──────────────────────────────
    print("\n[7/7] Business Metrics & Segment Report")
    report = build_segment_report(segments_df, features)
    print("\nSegment Business Report:")
    print(report[["Segment", "CustomerCount", "RevenueShare_pct",
                  "EstimatedCLV", "Priority"]].to_string(index=False))

    # ── Initial Live Data Seed ─────────────────────────────────
    print("\n[+] Seeding live data feed (30 days)...")
    append_live_data(days=30)

    # ── Summary ───────────────────────────────────────────────
    elapsed = round(time.time() - t0, 1)
    final_summary = {
        "pipeline_duration_seconds": elapsed,
        "raw_rows": summary["total_rows"],
        "clean_rows": len(df_clean),
        "unique_customers": features["Customer ID"].nunique(),
        "segments_found": segments_df["Segment"].nunique(),
        "classification_metrics": clf_metrics,
    }
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    with open(reports_dir / "pipeline_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {elapsed}s")
    print(f"  Segments found: {segments_df['Segment'].nunique()}")
    print(f"  Results in: reports/")
    print(f"\n  Next step: streamlit run dashboard/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
