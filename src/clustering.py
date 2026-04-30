"""
K-Means Clustering for Battery Degradation Patterns
=====================================================
Clusters batteries by engineered features, visualizes with PCA,
compares to actual health tiers, and logs to MLflow.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import mlflow

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "capacity_fade_rate",
    "resistance_growth_rate",
    "average_charge_discharge_ratio",
    "voltage_delta_trend",
    "cycle_count",
    "current_discharge_capacity",
    "current_SOH",
    "capacity_variance_last_10_cycles",
    "energy_efficiency",
]

TIER_ORDER = ["Good", "Moderate", "Weak"]
CLUSTER_COLORS = ["#58a6ff", "#d29922", "#f85149"]


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def run_clustering(
    features_path: str = "data/engineered_features.csv",
    output_dir: str = "data",
    models_dir: str = "models",
    mlruns_dir: str = "mlruns",
    n_clusters: int = 3,
) -> dict:
    """Run K-Means clustering, PCA visualization, and log to MLflow."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(features_path)
    print(f"[Clustering] Loaded {len(df)} samples from {features_path}")

    X = df[FEATURE_COLS].values

    # --- Scale ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- K-Means ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df["cluster"] = cluster_labels

    # --- PCA ---
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df["pca_x"] = pca_coords[:, 0]
    df["pca_y"] = pca_coords[:, 1]

    # --- Adjusted Rand Index ---
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(TIER_ORDER)
    true_labels = le.transform(df["health_tier"].values)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    print(f"  Adjusted Rand Index: {ari:.4f}")

    # --- Crosstab ---
    crosstab = pd.crosstab(df["cluster"], df["health_tier"], margins=True)
    print(f"\n  Cluster vs Health Tier Crosstab:")
    print(crosstab.to_string())

    # --- Centroid interpretation ---
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    centroid_df = pd.DataFrame(centroids_original, columns=FEATURE_COLS)
    centroid_df.index.name = "cluster"

    print(f"\n  Cluster Centroids (original scale):")
    print(centroid_df.round(4).to_string())

    # Interpret clusters
    interpretations = []
    for i in range(n_clusters):
        c = centroid_df.iloc[i]
        if c["current_SOH"] >= 90:
            desc = f"Cluster {i}: HIGH-HEALTH — High SOH ({c['current_SOH']:.1f}%), low fade rate, low resistance. Matches 'Good' tier."
        elif c["current_SOH"] >= 80:
            desc = f"Cluster {i}: MODERATE-DEGRADATION — Mid SOH ({c['current_SOH']:.1f}%), moderate fade and resistance growth. Matches 'Moderate' tier."
        else:
            desc = f"Cluster {i}: SEVERE-DEGRADATION — Low SOH ({c['current_SOH']:.1f}%), steep capacity fade, high resistance. Matches 'Weak' tier."
        interpretations.append(desc)
        print(f"  {desc}")

    # --- MLflow logging ---
    mlflow.set_tracking_uri(mlruns_dir)
    mlflow.set_experiment("Battery_SOH_Prediction")

    with mlflow.start_run(run_name="KMeans_Clustering"):
        mlflow.log_param("model_type", "KMeans")
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("n_init", 10)
        mlflow.log_param("features_used", ",".join(FEATURE_COLS))
        mlflow.log_metric("adjusted_rand_index", ari)
        mlflow.log_metric("inertia", kmeans.inertia_)
        mlflow.log_metric("pca_explained_variance_ratio_total", float(pca.explained_variance_ratio_.sum()))

        # PCA scatter plot (colored by cluster)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for i in range(n_clusters):
            mask = cluster_labels == i
            axes[0].scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                          c=CLUSTER_COLORS[i], label=f"Cluster {i}", alpha=0.7, s=40)
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        axes[0].set_title("K-Means Clusters (PCA)")
        axes[0].legend()

        tier_colors = {"Good": "#2ea043", "Moderate": "#d29922", "Weak": "#f85149"}
        for tier in TIER_ORDER:
            mask = df["health_tier"] == tier
            axes[1].scatter(pca_coords[mask, 0], pca_coords[mask, 1],
                          c=tier_colors[tier], label=tier, alpha=0.7, s=40)
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        axes[1].set_title("Actual Health Tiers (PCA)")
        axes[1].legend()

        plt.tight_layout()
        pca_path = os.path.join(models_dir, "cluster_pca_plot.png")
        plt.savefig(pca_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(pca_path)

        # Log crosstab as text
        crosstab_path = os.path.join(models_dir, "cluster_crosstab.txt")
        with open(crosstab_path, "w") as f:
            f.write(crosstab.to_string())
            f.write("\n\nCluster Interpretations:\n")
            for interp in interpretations:
                f.write(f"  {interp}\n")
        mlflow.log_artifact(crosstab_path)

    # --- Save outputs for dashboard ---
    cluster_path = os.path.join(output_dir, "cluster_results.csv")
    df.to_csv(cluster_path, index=False)
    print(f"\n[Clustering] Cluster results saved → {cluster_path}")

    cluster_meta = {
        "adjusted_rand_index": ari,
        "inertia": float(kmeans.inertia_),
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "centroids": centroid_df.to_dict(orient="index"),
        "interpretations": interpretations,
        "crosstab": crosstab.to_dict(),
    }
    meta_path = os.path.join(output_dir, "cluster_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(cluster_meta, f, indent=2, default=str)
    print(f"[Clustering] Metadata saved → {meta_path}")

    return cluster_meta


if __name__ == "__main__":
    run_clustering()
