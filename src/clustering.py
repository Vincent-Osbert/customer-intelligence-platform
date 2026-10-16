"""
clustering.py
-------------
Clusters customers using their predicted behavioral probability scores
(from classification.py) as input features.

Input:  DataFrame with columns [Customer ID, p_ishighvalue, p_isfrequent, p_ischurning]
Output: Same DataFrame with an added Segment column (string business name)

Why cluster on probability scores instead of raw features?
  Raw RFM values are skewed and live in different units.
  Probability scores are already on [0,1], carry the classifier's learned
  non-linear understanding of each behavior, and have a shared meaning:
    0.9 p_ishighvalue means the model is 90% confident this is a top spender.
  Clustering in this space means segments map directly to business behaviors.

Algorithm comparison:
  K-Means  — fast, interpretable, baseline
  GMM      — soft assignments (probability of belonging to each cluster),
             handles overlapping segments that K-Means forces to split
  We run both and compare Silhouette scores. GMM is preferred if its
  silhouette score is within 0.02 of K-Means, because soft assignments
  are more realistic (a customer is rarely 100% one segment).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

SCORE_COLUMNS = ["p_ishighvalue", "p_isfrequent", "p_ischurning"]
RANDOM_STATE = 42


def _select_k(X_scaled: np.ndarray, k_range: range) -> tuple[int, pd.DataFrame]:
    """
    Run elbow + silhouette analysis across k_range.
    Returns the recommended k and a results DataFrame.
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if k > 1 else 0.0
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})

    results = pd.DataFrame(rows)

    # Save elbow + silhouette plot
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(results["k"], results["inertia"], marker="o")
    axes[0].set(title="Elbow Curve — Where Does Inertia Stop Dropping Sharply?",
                xlabel="Number of Clusters (K)", ylabel="Inertia")
    axes[1].plot(results["k"], results["silhouette"], marker="o", color="green")
    axes[1].set(title="Silhouette Score — Higher = Better Defined Clusters",
                xlabel="Number of Clusters (K)", ylabel="Silhouette Score")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cluster_selection.png", dpi=150)
    plt.close(fig)

    # Pick K with highest silhouette
    best_k = int(results.loc[results["silhouette"].idxmax(), "k"])
    return best_k, results


def _name_segments(profiles: pd.DataFrame) -> dict[int, str]:
    """
    Map cluster IDs to business names using RELATIVE ranking of centroids.

    Rather than using fixed midpoint (0.5) thresholds, we rank clusters on
    each dimension. This works for any K because naming is relative:
    the cluster with highest p_ishighvalue + lowest p_ischurning = Champions.
    """
    n = len(profiles)
    rank_v = profiles["p_ishighvalue"].rank(ascending=True)   # higher = more valuable
    rank_f = profiles["p_isfrequent"].rank(ascending=True)    # higher = more frequent
    rank_c = profiles["p_ischurning"].rank(ascending=True)    # higher = more at-risk

    med = (n + 1) / 2   # median rank

    labels = {}
    for cluster_id in profiles.index:
        v, f, c = rank_v[cluster_id], rank_f[cluster_id], rank_c[cluster_id]

        if v == n and f >= med and c < med:
            name = "VIP Champions"
        elif v == n and c >= med:
            name = "High Value at Risk"
        elif v >= med and f == n and c < med:
            name = "VIP Champions"
        elif v >= med and c == n:
            name = "High Value at Risk"
        elif v >= med and f >= med and c < med:
            name = "Loyal Customers"
        elif f >= med and v < med and c < med:
            name = "Regular Engaged"
        elif c >= med and v < med:
            name = "Dormant / Lost"
        elif f < med and c < med and v < med:
            name = "Occasional Buyers"
        else:
            # Fallback
            dominant = max([("Value", v), ("Frequency", f), ("Churn Risk", c)], key=lambda x: x[1])
            name = f"Moderate ({dominant[0]})"
            
        labels[cluster_id] = name
    return labels


def run_clustering(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full clustering pipeline.

    1. Scale probability scores (mean=0, std=1) before clustering.
       Even though scores are on [0,1], their variances can differ —
       standardizing ensures no single dimension dominates the distance.
    2. Select K via elbow + silhouette.
    3. Run K-Means and GMM, compare silhouette.
    4. Assign business names to clusters.
    5. Save model, profiles, and labeled data.

    Returns the scores_df with an added 'Segment' column.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X = scores_df[SCORE_COLUMNS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 1: select K
    print("Selecting optimal number of clusters...")
    best_k, selection_results = _select_k(X_scaled, k_range=range(2, 8))
    print(f"  Recommended K = {best_k} (highest silhouette)")

    # Step 2: run K-Means
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    km_labels = kmeans.fit_predict(X_scaled)
    km_sil = silhouette_score(X_scaled, km_labels)

    # Step 3: run GMM
    gmm = GaussianMixture(n_components=best_k, random_state=RANDOM_STATE, n_init=5)
    gmm_labels = gmm.fit_predict(X_scaled)
    gmm_sil = silhouette_score(X_scaled, gmm_labels)

    print(f"  K-Means silhouette: {km_sil:.4f}")
    print(f"  GMM silhouette:     {gmm_sil:.4f}")

    # Prefer GMM if within 0.02 of K-Means (better for overlapping segments)
    if gmm_sil >= km_sil - 0.02:
        chosen_labels = gmm_labels
        chosen_model = gmm
        chosen_name = "GMM"
        print("  Selected: GMM (preferred for soft, overlapping segments)")
    else:
        chosen_labels = km_labels
        chosen_model = kmeans
        chosen_name = "KMeans"
        print("  Selected: K-Means (clearly better silhouette)")

    # Step 4: name clusters
    result = scores_df.copy()
    result["ClusterID"] = chosen_labels
    profiles = result.groupby("ClusterID")[SCORE_COLUMNS].mean()
    segment_names = _name_segments(profiles)
    result["Segment"] = result["ClusterID"].map(segment_names)

    # Step 5: save
    joblib.dump(chosen_model, MODELS_DIR / "clustering_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "cluster_scaler.joblib")

    profiles_full = result.groupby("Segment")[SCORE_COLUMNS].mean()
    profiles_full["CustomerCount"] = result.groupby("Segment")["Customer ID"].count()
    profiles_full.to_csv(REPORTS_DIR / "cluster_profiles.csv")

    result.to_csv(PROCESSED_DIR / "customer_segments.csv", index=False)

    algo_comparison = {
        "selected_algorithm": chosen_name,
        "best_k": best_k,
        "kmeans_silhouette": round(km_sil, 4),
        "gmm_silhouette": round(gmm_sil, 4),
    }
    with open(REPORTS_DIR / "clustering_metrics.json", "w") as f:
        json.dump(algo_comparison, f, indent=2)

    print(f"\nSegment distribution:")
    print(result["Segment"].value_counts().to_string())
    return result
