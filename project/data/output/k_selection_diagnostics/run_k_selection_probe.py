"""
K-selection diagnostics probe -- a controlled re-run of the K sweep in
03_kmeans_cluster_anomalies.py, used to explain why two silhouette values for
the same data appear in this repository (0.3662 vs 0.2198).

It reads the SAME live table 03 reads and copies 03's KMeans settings verbatim
(seed=42, maxIter=20, tol=1e-4, single init, local[14], 10g driver). For every
K in 2..8 it records FOUR numbers instead of one:

    wssse                   = model.summary.trainingCost               (what 03 prints)
    spark_squaredEuclidean  = ClusteringEvaluator(distanceMeasure="squaredEuclidean")
                                                                        (what 03 reports)
    sklearn_euclidean       = sklearn silhouette_score on the same labels
                                                                        (what 10_robust reports)
    n_clusters_found / cluster_sizes

Spark's ClusteringEvaluator accepts only "squaredEuclidean"; passing "euclidean"
raises IllegalArgumentException, which is why the Euclidean column is computed
with sklearn rather than with Spark.

It also audits the feature space for degenerate rows (zero vectors, duplicates),
because the first hypothesis for the 0.3662 / 0.2198 gap was that such rows were
inflating one of the metrics. They do not exist: minimum feature norm is 1.71.

Outputs (written next to this script):
    k_probe.csv          one row per K, columns as above
    probe_summary.json   input path, row count, argmax verdict, feature-norm audit

Run:  D:/Anaconda/python.exe project/data/output/k_selection_diagnostics/run_k_selection_probe.py
      (takes ~40 s; needs pyspark + Java; writes nothing outside this folder)
"""

import os
import sys
import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]  # .../CS5488project_Group101
INPUT_PATH = str(PROJECT_ROOT / "project" / "data" / "output" /
                 "if_results_rowkeyed" / "strict_anomalies_rowkeyed")
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    'log_miles', 'log_fare', 'log_tips', 'speed_mph', 'log_duration',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
]
K_RANGE = range(2, 9)

os.environ.setdefault("SPARK_LOCAL_DIRS", "D:/temp_spark")
spark = (SparkSession.builder
    .appName("KSandboxProbe")
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
spark.sparkContext.setLogLevel("ERROR")

print("=" * 72)
print("INPUT:", INPUT_PATH)
df = spark.read.parquet(INPUT_PATH)
n_total = df.count()
print(f"Loaded {n_total} anomaly records")
print("=" * 72)

# ---- features, verbatim from 03 -------------------------------------------
assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw")
df_assembled = assembler.transform(df)
scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                        withStd=True, withMean=True)
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)

# ---- zero-vector audit (computed on the SCALED features, which is what
# ---- ClusteringEvaluator actually sees) -----------------------------------
norm_udf = None
pdf_feat = df_scaled.select("features").toPandas()
X = np.vstack([np.asarray(v.toArray(), dtype=float) for v in pdf_feat["features"]])
norms = np.linalg.norm(X, axis=1)
print("\n--- feature-norm audit (standardized space) ---")
print(f"rows                      : {len(X)}")
print(f"exact zero vectors        : {int((norms == 0).sum())}")
print(f"norm < 1e-9               : {int((norms < 1e-9).sum())}")
print(f"norm < 1e-6               : {int((norms < 1e-6).sum())}")
print(f"norm min / median / max   : {norms.min():.6f} / {np.median(norms):.6f} / {norms.max():.6f}")
print(f"duplicate feature rows    : {len(X) - len(np.unique(X, axis=0))}")

# ---- per-K evaluation -----------------------------------------------------
ev_sq = ClusteringEvaluator(featuresCol="features", predictionCol="cluster",
                            metricName="silhouette", distanceMeasure="squaredEuclidean")

# Spark's ClusteringEvaluator only accepts "squaredEuclidean" (verified at run time:
# passing "euclidean" raises IllegalArgumentException), so the "euclidean" variant that
# 10_robust_clustering_rowkeyed.py uses via sklearn is computed here with sklearn.

from sklearn.metrics import silhouette_score

rows = []
models = {}
for k in K_RANGE:
    km = KMeans(featuresCol="features", predictionCol="cluster",
                k=k, seed=42, maxIter=20, tol=1e-4)
    model = km.fit(df_scaled)
    preds = model.transform(df_scaled)
    cost = model.summary.trainingCost

    sil_sq = ev_sq.evaluate(preds)

    labels = np.array([r[0] for r in
                       preds.select("cluster").toPandas().itertuples(index=False, name=None)])
    sil_sk = float(silhouette_score(X, labels, metric="euclidean"))

    models[k] = model
    rows.append({"k": k,
                 "wssse": round(cost, 2),
                 "spark_squaredEuclidean": round(float(sil_sq), 4),
                 "sklearn_euclidean": round(sil_sk, 4),
                 "n_clusters_found": int(len(np.unique(labels))),
                 "cluster_sizes": "/".join(str(int((labels == c).sum()))
                                           for c in np.unique(labels))})
    print(f"K={k}: WSSSE={cost:11.2f} | spark_sq={sil_sq:7.4f} | "
          f"sklearn_eu={sil_sk:7.4f} | sizes={rows[-1]['cluster_sizes']}")

res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / "k_probe.csv", index=False)

# ---- what each metric would choose ---------------------------------------
print("\n" + "=" * 72)
verdict = {}
for col in ["spark_squaredEuclidean", "sklearn_euclidean"]:
    best = int(res.loc[res[col].idxmax(), "k"])
    verdict[col] = best
    print(f"argmax({col:24s}) = K={best}")
print("=" * 72)

# ---- where do near-origin rows land, in the model that 03 would pick -----
k_03 = verdict["spark_squaredEuclidean"]
lab_03 = np.array([r[0] for r in models[k_03].transform(df_scaled)
                   .select("cluster").toPandas().itertuples(index=False, name=None)])
near = norms < 1e-6
if near.any():
    print(f"\nnear-origin rows (<1e-6) in K={k_03} model: {int(near.sum())}")
    for c in np.unique(lab_03):
        sel = (lab_03 == c)
        print(f"  cluster {c}: n={int(sel.sum())}, near-origin={int((near & sel).sum())}")
else:
    print("\nno near-origin rows found -- the zero-vector hypothesis is NOT supported")

json.dump({"input": INPUT_PATH,
           "n_rows": int(n_total),
           "verdict_argmax": verdict,
           "zero_vectors_exact": int((norms == 0).sum()),
           "norm_min": float(norms.min()),
           "duplicate_feature_rows": int(len(X) - len(np.unique(X, axis=0)))},
          open(OUT_DIR / "probe_summary.json", "w"), indent=2)

print(f"\nwrote: {OUT_DIR / 'k_probe.csv'}")
print(f"wrote: {OUT_DIR / 'probe_summary.json'}")
spark.stop()
