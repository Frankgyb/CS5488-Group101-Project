"""
Location: project/code/ml/analysis/09_eda_clustering.py | PROJECT_ROOT depth: 3
eda_clustering.py - Exploratory Data Analysis for anomaly clustering
Uses relative paths based on project root.
Produces data/output/graph/{feature_distributions,outlierScore_distribution,pca_projection}.png.
NOTE: the PCA scatter is coloured by `outlierScore`, NOT by cluster - the "detached
slender strip" is a zero-tip sub-group inside long_low_tip (PC2 < -1.5: short_high_tip
16.7% vs long_low_tip 7.7%), so this figure is NOT independent evidence for K=2.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

# ------------------------------
# 1. Set up relative paths
# ------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../project

DATA_INPUT = PROJECT_ROOT / "data" / "output" / "if_results_rowkeyed" / "strict_anomalies_rowkeyed"

OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "graph"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------
# 2. Load data
# ------------------------------
anom = pd.read_parquet(DATA_INPUT)
print(f"Loaded {len(anom)} anomaly records from {DATA_INPUT}")

FEATURE_COLS = [
    'log_miles', 'log_fare', 'log_tips', 'speed_mph', 'log_duration',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
]

# ------------------------------
# 3. Feature skewness
# ------------------------------
print("=== Feature Skewness ===")
skew = anom[FEATURE_COLS].skew().sort_values(ascending=False)
print(skew)


fig, axes = plt.subplots(3, 3, figsize=(12, 10))
axes = axes.flatten()
for i, col in enumerate(FEATURE_COLS):
    axes[i].hist(anom[col], bins=50, edgecolor='k', alpha=0.7)
    axes[i].set_title(col)
plt.tight_layout()
plt.suptitle("Feature Distributions (Standardized Scale)", y=1.02)
plt.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=150)
plt.show()

# ------------------------------
# 4. outlierScore distribution
# ------------------------------
print("=== Outlier Score Statistics ===")
print(anom['outlierScore'].describe())

plt.figure(figsize=(8, 4))
plt.hist(anom['outlierScore'], bins=80, edgecolor='k', alpha=0.7)
plt.axvline(x=anom['outlierScore'].median(), color='red', linestyle='--', label='Median')
plt.xlabel('Outlier Score')
plt.ylabel('Frequency')
plt.title('Distribution of outlierScore among anomalies')
plt.legend()
plt.savefig(OUTPUT_DIR / "outlierScore_distribution.png", dpi=150)
plt.show()

# ------------------------------
# 5. PCA 2D projection
# ------------------------------
print("=== PCA 2D Projection ===")
X = anom[FEATURE_COLS].values
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
explained_var = pca.explained_variance_ratio_.sum()
print(f"Explained variance ratio (2 components): {explained_var:.3f}")

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=anom['outlierScore'],
                      cmap='viridis', s=5, alpha=0.6)
plt.colorbar(scatter, label='Outlier Score')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection of Anomalies (colored by outlierScore)')
plt.savefig(OUTPUT_DIR / "pca_projection.png", dpi=150)
plt.show()

# ------------------------------
# 6. Decision guidance
# ------------------------------
print("=== Decision Guidance ===")
print("- If PCA shows distinct blobs: use KMeans with silhouette score.")
print("- If no clear structure: try DBSCAN or accept that anomalies are homogeneous.")
print("- outlierScore spread is wide enough? Check histogram above.")