"""
Location: project/code/ml/out_of_use/code_legacy_shap/07_shap_cluster_analysis.py | PROJECT_ROOT = parents[4]
SUPERSEDED by ml/analysis/07b_shap_cluster_analysis_rowkeyed.py; its inputs (kmeans_anomaly_clusters/, if_results_10pct_of_10pct/) are archived under data/output/out_of_use/_outputs/, so this script cannot run as-is.
shap_cluster_analysis.py - SHAP analysis per cluster for KMeans-clustered anomalies.
Loads clustered anomaly data and corresponding SHAP values,
computes mean absolute SHAP per cluster, identifies top-3 driving features,
and saves bar plots and summary CSV.

Inputs:
  - Clustered anomalies: data/output/kmeans_anomaly_clusters/strict_anomalies_clustered
  - SHAP values:         data/output/if_results_10pct_of_10pct/shap_strict_anomalies.csv
Outputs (all in data/output/kmeans_anomaly_clusters/):
  - shap_cluster_summary.csv   : mean SHAP per cluster (absolute and signed)
  - shap_top3_per_cluster.csv  : top-3 features per cluster
  - shap_bar_{cluster}.png     : bar plot for each cluster
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. Relative path setup
# ------------------------------------------------------------
# 2026-11-22: this legacy script keeps its ORIGINAL inputs/outputs (old 02 table,
# old 03 clusters) but they were archived; every path now carries the
# out_of_use/_outputs/ prefix so the script resolves without touching the live tree.
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # .../project
VOID = PROJECT_ROOT / "data" / "output" / "out_of_use" / "_outputs"

CLUSTERED_PATH = VOID / "kmeans_anomaly_clusters" / "strict_anomalies_clustered"
SHAP_PATH = VOID / "if_results_10pct_of_10pct" / "shap_strict_anomalies.csv"
OUTPUT_DIR = VOID / "kmeans_anomaly_clusters"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------
print("Loading clustered anomaly data...")
df_clustered = pd.read_parquet(CLUSTERED_PATH)
print(f"Clustered data shape: {df_clustered.shape}")

print("Loading SHAP values...")
df_shap = pd.read_csv(SHAP_PATH)
print(f"SHAP data shape: {df_shap.shape}")

# Align by row count (assume SHAP corresponds to first N rows of anomalies)
if len(df_clustered) != len(df_shap):
    print(f"Row count mismatch. Truncating clustered data to {len(df_shap)} rows.")
    df_clustered = df_clustered.iloc[:len(df_shap)].reset_index(drop=True)
    df_shap = df_shap.reset_index(drop=True)

# Add cluster labels to SHAP dataframe
df_shap['cluster'] = df_clustered['cluster'].values
print(f"Cluster distribution:\n{df_shap['cluster'].value_counts().sort_index()}")

# Feature columns
FEATURE_COLS = [
    'log_miles', 'log_fare', 'log_tips', 'speed_mph', 'log_duration',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
]
SHAP_COLS = [f'shap_{col}' for col in FEATURE_COLS]

# ------------------------------------------------------------
# 3. Compute mean SHAP per cluster (absolute and signed)
# ------------------------------------------------------------
cluster_groups = df_shap.groupby('cluster')

# Mean absolute SHAP (magnitude of influence)
mean_abs_shap = cluster_groups[SHAP_COLS].apply(lambda x: x.abs().mean()).reset_index()
mean_abs_shap.columns = ['cluster'] + [f'abs_{col}' for col in SHAP_COLS]

# Mean signed SHAP (direction of influence)
mean_signed_shap = cluster_groups[SHAP_COLS].mean().reset_index()
mean_signed_shap.columns = ['cluster'] + [f'mean_{col}' for col in SHAP_COLS]

# Merge
shap_summary = mean_abs_shap.merge(mean_signed_shap, on='cluster')
shap_summary.to_csv(OUTPUT_DIR / "shap_cluster_summary.csv", index=False)
print("\nSaved shap_cluster_summary.csv")

# ------------------------------------------------------------
# 4. Extract top-3 driving features per cluster
# ------------------------------------------------------------
top3_records = []
clusters = sorted(df_shap['cluster'].unique())

for cl in clusters:
    sub = df_shap[df_shap['cluster'] == cl]
    mean_abs = sub[SHAP_COLS].abs().mean()
    top3 = mean_abs.sort_values(ascending=False).head(3)
    for feat in top3.index:
        base_feat = feat.replace('shap_', '')
        signed_mean = sub[feat].mean()
        top3_records.append({
            'cluster': cl,
            'feature': base_feat,
            'mean_abs_shap': round(top3[feat], 4),
            'mean_signed_shap': round(signed_mean, 4)
        })

top3_df = pd.DataFrame(top3_records)
top3_df.to_csv(OUTPUT_DIR / "shap_top3_per_cluster.csv", index=False)
print("Saved shap_top3_per_cluster.csv")
print("\nTop-3 features per cluster:")
print(top3_df.to_string(index=False))

# ------------------------------------------------------------
# 5. Bar plots for each cluster
# ------------------------------------------------------------
for cl in clusters:
    sub = df_shap[df_shap['cluster'] == cl]
    mean_shap = sub[SHAP_COLS].mean().sort_values()
    colors = ['red' if v < 0 else 'blue' for v in mean_shap.values]

    plt.figure(figsize=(8, 5))
    mean_shap.plot(kind='barh', color=colors)
    plt.axvline(x=0, color='black', linewidth=0.5)
    plt.title(f'Mean SHAP Values – Cluster {cl} ({"LowValueHighTip" if cl==0 else "HighValueZeroTip"})')
    plt.xlabel('Mean SHAP value')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"shap_bar_cluster_{cl}.png", dpi=150)
    plt.show()
    print(f"Saved shap_bar_cluster_{cl}.png")

print("\nAnalysis complete. All outputs in:", OUTPUT_DIR)