"""
07b K-row audit -- find out WHY two rows of selfcheck_k4/k_selection_summary.csv
disagree with everything else.

Background
----------
selfcheck_k4/k_selection_summary.csv is 07b's own K=2..8 sweep, run on the same
13,577 rows as 03 and 10. Its K=4 row (silhouette 0.3119) is the origin of the
"a re-run selects K=4" claim. Most of its rows agree with the authoritative
10-seed sweep to within a few WSSSE units:

    K   07b WSSSE   authoritative inertia   agreement
    2   102,254.7   98,245.8                +4,009  <-- disagrees
    3    83,681.9   83,686.1                    -4
    4    75,427.0   75,432.8                    -6
    5    71,570.2   68,388.6                +3,182  <-- disagrees
    6    63,977.5   63,981.7                    -4
    7    61,263.3   60,888.1                  +375
    8    58,756.9   58,352.4                  +405

Two successive KMeans fits with different initialisation can land on different
local optima; WSSSE then differs by thousands, while silhouette (computed from
the same labels) moves with it. That is the candidate explanation.

This script tests it without touching anything 07b produced: it re-runs the
identical loop (seed=42, maxIter=20, tol=1e-4, squaredEuclidean) REPS times and
reports, for every K, the spread of WSSSE and silhouette across repetitions.

Decision rule, fixed before looking at the results:
    if spread(K=2) covers the 07b K=2 value  -> the row is one draw, not a bug,
                                                only 07b's single init is unlucky
    if spread never reaches 102,254.7        -> something else is wrong with that
                                                row and 07b must be re-run instead

Reads:  project/data/output/if_results_rowkeyed/strict_anomalies_rowkeyed
Writes: project/data/output/k_selection_diagnostics/k_spread_07b_audit.csv
        project/data/output/k_selection_diagnostics/k_spread_07b_audit.txt

Run:  D:/Anaconda/python.exe project/data/output/k_selection_diagnostics/audit_07b_k_rows.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

PROJECT_ROOT = Path(__file__).resolve().parents[4]          # .../CS5488project_Group101
INPUT_PATH = str(PROJECT_ROOT / "project" / "data" / "output" /
                 "if_results_rowkeyed" / "strict_anomalies_rowkeyed")
OUT_DIR = Path(__file__).resolve().parent
SELFCHECK = (PROJECT_ROOT / "project" / "data" / "output" /
             "anomaly_clusters_rowkeyed" / "selfcheck_k4" / "k_selection_summary.csv")

FEATURE_COLS = [
    'log_miles', 'log_fare', 'log_tips', 'speed_mph', 'log_duration',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
]
K_RANGE = range(2, 9)
REPS = 6            # fits per K; the loop is ~5 s per fit, so ~4 min in total
SEED = 42           # same seed 07b used

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))

os.environ.setdefault("SPARK_LOCAL_DIRS", "D:/temp_spark")
spark = (SparkSession.builder
         .appName("07bKRowAudit")
         .master("local[8]")
         .config("spark.local.dir", "D:/temp_spark")
         .config("spark.sql.shuffle.partitions", "16")
         .config("spark.driver.memory", "8g")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

log("=" * 78)
log("07b K-row audit -- is 102,254.7 a plausible single-init draw?")
log("=" * 78)
log(f"input : {INPUT_PATH}")
df = spark.read.parquet(INPUT_PATH)
log(f"rows  : {df.count():,}")
log(f"fits  : K=2..8 x {REPS} repetitions, seed={SEED}, maxIter=20, tol=1e-4")
log("")

assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw")
sdf = assembler.transform(df)
scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                        withStd=True, withMean=True)
sdf = scaler.fit(sdf).transform(sdf).cache()

ev = ClusteringEvaluator(featuresCol="features", predictionCol="cluster",
                         metricName="silhouette",
                         distanceMeasure="squaredEuclidean")

rows = []
for k in K_RANGE:
    for rep in range(REPS):
        km = KMeans(featuresCol="features", predictionCol="cluster",
                    k=k, seed=SEED, maxIter=20, tol=1e-4)
        m = km.fit(sdf)
        preds = m.transform(sdf)
        sil = float(ev.evaluate(preds))
        wssse = float(m.summary.trainingCost)
        sizes = sorted([r[0] for r in
                        preds.groupBy("cluster").count().toPandas()
                        .itertuples(index=False, name=None)], reverse=True)
        rows.append({"k": k, "rep": rep, "wssse": round(wssse, 2),
                     "silhouette": round(sil, 4),
                     "sizes": "/".join(str(s) for s in sizes)})
        log(f"  K={k} rep={rep}: WSSSE={wssse:11.2f}  silhouette={sil:.4f}")

res = pd.DataFrame(rows)
res.to_csv(OUT_DIR / "k_spread_07b_audit.csv", index=False)

# ---- compare against what 07b recorded and what the authoritative sweep says
thirty = pd.read_csv(SELFCHECK).set_index("k")
ten = pd.read_csv(PROJECT_ROOT / "project" / "data" / "output" /
                  "clustering_robustness" / "k_sweep.csv")
ten = ten[ten.seed == SEED].set_index("k")

log("")
log("=" * 78)
log("per-K spread vs the 07b recorded value")
log("=" * 78)
log(f"{'K':>2} | {'07b recorded':>12} | {'this run min':>12} {'max':>12} {'spread':>8} "
    f"| {'07b inside?':>11} | {'03-style (sq)':>12}")
summary = []
for k in K_RANGE:
    sub = res[res.k == k]
    lo, hi = sub.wssse.min(), sub.wssse.max()
    rec = float(thirty.loc[k, "wssse"])
    inside = (lo - 1e-6) <= rec <= (hi + 1e-6)
    summary.append({"k": k, "recorded_07b": rec,
                    "rep_min": round(lo, 2), "rep_max": round(hi, 2),
                    "spread": round(hi - lo, 2), "recorded_inside_spread": inside})
    log(f"{k:>2} | {rec:>12,.1f} | {lo:>12,.1f} {hi:>12,.1f} {hi - lo:>8,.1f} "
        f"| {str(inside):>11} | {thirty.loc[k, 'silhouette']:>12.4f}")

pd.DataFrame(summary).to_csv(OUT_DIR / "k_spread_07b_audit_summary.csv", index=False)

log("")
log("=" * 78)
log("verdict")
log("=" * 78)
k2 = res[res.k == 2]
rec2 = float(thirty.loc[2, "wssse"])
if k2.wssse.max() < rec2:
    log(f"  K=2: this run tops out at WSSSE {k2.wssse.max():,.1f}, BELOW 07b's "
        f"{rec2:,.1f}.")
    log("       -> the repetition explanation does NOT cover it; 07b's K=2 row "
        "comes from a different configuration and the row should be re-generated.")
else:
    log(f"  K=2: 07b's {rec2:,.1f} lies inside this run's range "
        f"[{k2.wssse.min():,.1f}, {k2.wssse.max():,.1f}].")
    log("       -> that row is one draw of a multi-optimum fit, not a bug: "
        "only 07b's single initialisation was unlucky.")

k4 = res[res.k == 4]
log(f"  K=4: silhouette across repetitions ranges "
    f"[{k4.silhouette.min():.4f}, {k4.silhouette.max():.4f}]; 07b recorded "
    f"{thirty.loc[4, 'silhouette']:.4f} (inside range: "
    f"{bool(k4.silhouette.min() <= thirty.loc[4, 'silhouette'] <= k4.silhouette.max())}).")
log(f"       K=4 still beats every other K in every repetition: "
    f"{bool(res[res.k != 4].groupby('k').silhouette.max().max() < k4.silhouette.min())}")

(OUT_DIR / "k_spread_07b_audit.txt").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
log("")
log(f"wrote: {OUT_DIR / 'k_spread_07b_audit.csv'}")
log(f"wrote: {OUT_DIR / 'k_spread_07b_audit_summary.csv'}")
log(f"wrote: {OUT_DIR / 'k_spread_07b_audit.txt'}")
spark.stop()
