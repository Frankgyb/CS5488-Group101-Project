# Location: project/code/ml/analysis/08_feature_viz.py | PROJECT_ROOT depth: 3
# (formerly project/code/ml/feature_viz.py)
import os
from pyspark.sql import SparkSession
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============ Configuration ============
# Location: ml/analysis/08_feature_viz.py -> 3 levels up = project/
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
# DEPRECATED 2026-11-22: reads the legacy 03 K=2 labels (NOT row_uid-keyed, and the
# numbering is the reverse of the robust clustering), so the plots drawn here show
# raw cluster ids. That is acceptable for feature DISTRIBUTIONS but the ids must not
# be quoted in the report; write long_low_tip / short_high_tip instead.
# Path updated 2026-11-22: the parquet moved to data/output/anomaly_clusters_rowkeyed/.
# For a keyed version, join data/output/clustering_robustness/row_uid_cluster_robust.parquet
# on row_uid instead.
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "output", "anomaly_clusters_rowkeyed", "strict_anomalies_clustered")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output", "graph")

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "log_miles", "log_fare", "log_tips", "speed_mph", "log_duration",
    "hour_sin", "hour_cos", "day_sin", "day_cos"
]
KEY_FEATS = ["log_miles", "log_fare", "log_tips", "speed_mph", "log_duration"]

# ============ Spark Session ============
spark = (SparkSession.builder
         .appName("ClusterFeatureViz")
         .master("local[*]")
         .config("spark.driver.memory", "12g")
         .config("spark.driver.maxResultSize", "8g")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .getOrCreate())

# ============ 1. Load clustered anomalies ============
print("Loading clustered anomalies...")
df_spark = spark.read.parquet(INPUT_PATH)

# Convert to Pandas (only 13,577 rows – no need for subsampling)
pdf = df_spark.select(*FEATURE_COLS, "cluster").toPandas()
pair_pdf = df_spark.select(*KEY_FEATS, "cluster").toPandas()
print(f"Loaded {len(pdf):,} rows with {pdf['cluster'].nunique()} clusters")

# ============ 2. Set Plot Style ============
sns.set_style("whitegrid")

# ============ 3. Histograms + Density Plots (by cluster) ============
print("Generating histograms and density plots by cluster...")
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()
for idx, col in enumerate(FEATURE_COLS):
    ax = axes[idx]
    sns.histplot(data=pdf, x=col, hue='cluster', kde=True, bins=50,
                 palette='Set2', edgecolor='none', alpha=0.6, ax=ax)
    ax.set_title(f"{col} Distribution by Cluster", fontsize=14)
    ax.set_xlabel('')
plt.tight_layout()
out_path1 = os.path.join(OUTPUT_DIR, "viz_hist_density.png")
plt.savefig(out_path1, dpi=150)
print(f"Histograms saved to: {out_path1}")
plt.close()

# ============ 4. Boxplots (by cluster) ============
print("Generating boxplots by cluster...")
fig2, axes2 = plt.subplots(3, 3, figsize=(18, 12))
axes2 = axes2.flatten()
for idx, col in enumerate(FEATURE_COLS):
    ax = axes2[idx]
    sns.boxplot(data=pdf, x='cluster', y=col, palette='Set2', ax=ax, showfliers=True)
    ax.set_title(f"{col} Boxplot by Cluster", fontsize=14)
plt.tight_layout()
out_path2 = os.path.join(OUTPUT_DIR, "viz_boxplots.png")
plt.savefig(out_path2, dpi=150)
print(f"Boxplots saved to: {out_path2}")
plt.close()

# ============ 5. Pairplot (by cluster) ============
print("Generating pairplot by cluster...")
g = sns.pairplot(pair_pdf, hue='cluster', diag_kind='kde',
                 palette='Set2', plot_kws={'alpha':0.3, 's':2})
out_path3 = os.path.join(OUTPUT_DIR, "viz_pairplot.png")
plt.savefig(out_path3, dpi=150)
print(f"Pairplot saved to: {out_path3}")
plt.close()

spark.stop()
print("Visualization completed!")
