"""
Location: project/code/ml/pipeline/03_kmeans_cluster_anomalies.py | PROJECT_ROOT depth: 3
kmeans_cluster_anomaly.py - KMeans clustering on Isolation Forest anomalies.
Selects best K (2-8) by silhouette, labels anomalies, saves results.

Input : data/output/if_results_rowkeyed/strict_anomalies_rowkeyed  (row_uid-keyed)
Output: data/output/anomaly_clusters_rowkeyed/   (LIVE location, 2026-11-22)
        data/output/out_of_use/_outputs/anomaly_clusters_positional_jsonvalid/
            k_selection_summary.csv   (void: inflated silhouette, see NOTE below)

NOTE (2026-11-22) -- this script is a LEGACY / DIAGNOSTIC entry point.
  Its silhouette column is not a reliable model-selection signal: the original
  run (log: data/log/kmeans_cluster_anomalies_output.txt, which reads the OLD
  non-rowkeyed `if_results_10pct_of_10pct/strict_anomalies`) reported
  K=2 -> 0.3662, while the SAME K on the rowkeyed anomalies gives 0.2731 with
  an almost identical WSSSE (98,238 vs 102,255). Same n, same inertia, wildly
  different silhouette => the old 0.3662 column is inflated (zero vectors /
  degenerate clusters), not an initialisation artefact.
  The AUTHORITATIVE K choice is analysis/10_robust_clustering_rowkeyed.py
  (10 seeds x n_init=50): K=2, silhouette 0.2198 +/- 0.0000, bootstrap ARI 0.96.
"""

from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import pandas as pd

# ------------------------------------------------------------
# 1. Spark session
# ------------------------------------------------------------
spark = (SparkSession.builder
    .appName("KMeansAnomalyClusters")
    .master("local[14]")
    .config("spark.local.dir", "D:/temp_spark")
    .config("spark.sql.shuffle.partitions", "30")
    .config("spark.default.parallelism", "30")
    .config("spark.driver.memory", "10g")
    .config("spark.driver.memoryOverhead", "2g")
    .config("spark.memory.fraction", "0.6")
    .config("spark.memory.storageFraction", "0.3")
    .config("spark.memory.offHeap.enabled", "false")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
    .getOrCreate()
)

# ------------------------------------------------------------
# 2. Paths (relative to project root; script lives in code/ml/)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../project

INPUT_PATH = str(PROJECT_ROOT / "data" / "output" /
                 "if_results_rowkeyed" / "strict_anomalies_rowkeyed")

# LIVE output (moved out of out_of_use/ on 2026-11-22): 10_robust_clustering_rowkeyed.py
# and add_consistency_features_v2.py both READ this directory, so it is an input of
# live code and must not sit in the deprecated tree.
OUTPUT_DIR = (PROJECT_ROOT / "data" / "output" / "anomaly_clusters_rowkeyed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The K-sweep table goes to the deprecated tree, NOT above. Its silhouette column is
# the inflated one described in the module docstring (0.3662 at K=2 against 0.2198
# for the same data), so it must not resurface in the live directory on a re-run.
VOID_DIR = (PROJECT_ROOT / "data" / "output" / "out_of_use" / "_outputs" /
            "anomaly_clusters_positional_jsonvalid")
VOID_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 3. Read anomaly data
# ------------------------------------------------------------
df = spark.read.parquet(INPUT_PATH)
n_total = df.count()
print(f"Loaded {n_total} anomaly records from: {INPUT_PATH}")

FEATURE_COLS = [
    'log_miles', 'log_fare', 'log_tips', 'speed_mph', 'log_duration',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
]

# ------------------------------------------------------------
# 4. Assemble features and standardize
# ------------------------------------------------------------
assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw")
df_assembled = assembler.transform(df)

scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                        withStd=True, withMean=True)
df_scaled = scaler.fit(df_assembled).transform(df_assembled)

# ------------------------------------------------------------
# 5. Select best K by silhouette score
# ------------------------------------------------------------
K_RANGE = range(2, 9)
evaluator = ClusteringEvaluator(
    featuresCol="features", predictionCol="cluster",
    metricName="silhouette", distanceMeasure="squaredEuclidean"
)

rows = []
best_k, best_sil, best_model = None, -1.0, None

for k in K_RANGE:
    km = KMeans(featuresCol="features", predictionCol="cluster",
                k=k, seed=42, maxIter=20, tol=1e-4)
    model = km.fit(df_scaled)
    preds = model.transform(df_scaled)
    sil = evaluator.evaluate(preds)
    cost = model.summary.trainingCost
    rows.append((k, round(cost, 2), round(sil, 4)))
    print(f"K={k}: WSSSE={cost:.2f}, silhouette={sil:.4f}")

    if sil > best_sil:
        best_sil, best_k, best_model = sil, k, model

print(f"\nBest K = {best_k} (silhouette = {best_sil:.4f})")

summary_pd = pd.DataFrame(rows, columns=["k", "wssse", "silhouette"])
summary_pd["is_best"] = summary_pd["k"] == best_k
summary_pd.to_csv(str(VOID_DIR / "k_selection_summary.csv"), index=False)

# ------------------------------------------------------------
# 6. Final labels and cluster sizes
# ------------------------------------------------------------
df_clustered = best_model.transform(df_scaled)

size_pd = (df_clustered.groupBy("cluster").count()
           .orderBy("cluster").toPandas())
size_pd["ratio"] = (size_pd["count"] / n_total).round(4)
print("\nCluster sizes:")
print(size_pd.to_string(index=False))
size_pd.to_csv(str(OUTPUT_DIR / "cluster_sizes.csv"), index=False)

# ------------------------------------------------------------
# 7. Cluster centers (standardized scale)
# ------------------------------------------------------------
centers_pd = pd.DataFrame([list(c) for c in best_model.clusterCenters()],
                          columns=FEATURE_COLS)
centers_pd.insert(0, "cluster", range(best_k))
print("\nCluster centers (standardized scale):")
print(centers_pd.round(4).to_string(index=False))
centers_pd.to_csv(str(OUTPUT_DIR / "cluster_centers.csv"), index=False)

# ------------------------------------------------------------
# 8. Save clustered anomalies (original columns + cluster)
# ------------------------------------------------------------
output_cols = df.columns + ["cluster"]
out_path = str(OUTPUT_DIR / "strict_anomalies_clustered")
df_clustered.select(output_cols).write.mode("overwrite").parquet(out_path)
print(f"\nClustered anomalies saved to: {out_path}")

# ------------------------------------------------------------
# 9. Stop Spark
# ------------------------------------------------------------
spark.stop()
print("Spark session stopped.")