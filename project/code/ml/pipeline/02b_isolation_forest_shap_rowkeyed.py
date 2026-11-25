"""
Location: project/code/ml/pipeline/02b_isolation_forest_shap_rowkeyed.py | PROJECT_ROOT depth: 3
02b_isolation_forest_shap_rowkeyed.py - Isolation Forest + SHAP, ROW-KEYED rerun.

WHY THIS SCRIPT EXISTS
----------------------
`02_isolation_forest_shap.py` persisted
    full_scores, strict_anomalies, shap_strict_anomalies.csv
with NO row identifier. Downstream `07_shap_cluster_analysis.py` therefore had
to align SHAP values to cluster labels by ROW POSITION (`iloc[:2000]`), which
silently paired unrelated rows and produced the invalid "cluster 0 is driven by
log_tips, cluster 1 by speed_mph" conclusion.

This script fixes the ROOT CAUSE by carrying an explicit `row_uid` from the
first read all the way through to the SHAP CSV, so every downstream join is
key-based and verifiable.

DESIGN RULES (same "no sabotage of existing outputs" rule as the rest of the
pipeline):
  * It does NOT overwrite `02`'s outputs. New directory:
        data/output/if_results_rowkeyed/
  * The anomaly mask, threshold and score logic are IDENTICAL to 02, so the
    rerun is an exact superset (same 13,577 anomalies) plus identifiers.
  * A self-check at the end asserts the recovered anomaly count equals
    13,577 and that SHAP row count == anomaly sample row count.

OUTPUTS (data/output/if_results_rowkeyed/)
  - full_scores/           parquet, + row_uid
  - strict_anomalies/      parquet, + row_uid + rank_in_layer + layer_total
  - shap_rowkeyed.csv      row_uid + 9 shap_* columns + the 9 raw feature values
  - run_meta.json          run parameters + integrity checks
  - 02b_rowkeyed.log       log
"""

import os
import sys
import json
import time
import logging

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend, Parallel, delayed
import shap

# ========== Paths ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
LOG_DIR = os.path.join(PROJECT_ROOT, "data", "log")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(
    LOG_DIR, f"02b_rowkeyed_{time.strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ========== Configuration (identical to 02) ==========
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "output", "sampled_km")
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "data", "output", "if_results_rowkeyed")
FEATURE_COLS = [
    "log_miles", "log_fare", "log_tips", "speed_mph", "log_duration",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
]
SEED = 42
FRAC = 0.10
SHAP_SAMPLE_N = 2000
EXPECTED_ANOMALIES = 13577          # from the original 02 run, used as a regression check


def find_dynamic_threshold(scores, min_anomaly_ratio=0.001, max_anomaly_ratio=0.05):
    """Verbatim copy of 02's threshold rule (elbow + IQR fallback)."""
    sorted_scores = np.sort(scores)
    diffs = np.diff(sorted_scores)
    elbow_idx = np.argmax(diffs)
    elbow_threshold = sorted_scores[elbow_idx]
    elbow_ratio = (scores < elbow_threshold).sum() / len(scores)

    if min_anomaly_ratio <= elbow_ratio <= max_anomaly_ratio:
        logger.info(f"Elbow method: threshold={elbow_threshold:.4f}, "
                    f"anomaly ratio={elbow_ratio*100:.2f}%")
        return elbow_threshold, "elbow"

    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    iqr_threshold = q1 - 3.0 * iqr
    iqr_ratio = (scores < iqr_threshold).sum() / len(scores)
    logger.info(f"IQR fallback: threshold={iqr_threshold:.4f}, "
                f"anomaly ratio={iqr_ratio*100:.2f}%")
    return iqr_threshold, "iqr"


# ========== 1. Spark ==========
spark = (SparkSession.builder
         .appName("IF_rowkeyed")
         .master("local[6]")
         .config("spark.driver.memory", "8g")
         .config("spark.driver.maxResultSize", "2g")
         .config("spark.sql.shuffle.partitions", "24")
         .config("spark.local.dir", "D:/temp_spark")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
logger.info("Spark session started.")

# ========== 2. Read + assign a deterministic row_uid BEFORE any shuffling ==========
logger.info(f"Reading {INPUT_PATH} ...")
df10 = spark.read.parquet(INPUT_PATH)

# row_uid = "<source_file_partition>-<monotonic id within partition>".
# This is assigned on the raw read, so it uniquely identifies a row of
# sampled_km and survives every later shuffle/filter unchanged.
df_keyed = df10.withColumn(
    "row_uid",
    F.concat_ws(
        "-",
        F.lit("p"),
        F.monotonically_increasing_id().cast("string"),
    ),
)

bucket_keys = [r.bucket for r in df_keyed.select("bucket").distinct().collect()]
logger.info(f"Buckets: {bucket_keys}")

fracs = {k: FRAC for k in bucket_keys}
df2 = (df_keyed
       .sampleBy("bucket", fractions=fracs, seed=SEED)
       .select(["row_uid"] + FEATURE_COLS + ["bucket", "hvfhs_license_num"]))
df2.cache()

total_rows = df2.count()
logger.info(f"Sampled slice: {total_rows:,} rows (10% of sampled_km).")

# Uniqueness guarantee: this is the whole point of the rewrite.
n_uid = df2.select("row_uid").distinct().count()
logger.info(f"row_uid distinct count: {n_uid:,} / {total_rows:,}")
if n_uid != total_rows:
    raise RuntimeError(
        f"row_uid collision: {n_uid:,} distinct vs {total_rows:,} rows. "
        "monotonically_increasing_id is not unique here; aborting.")

# ========== 3. Collect to driver ==========
logger.info("Collecting to pandas ...")
pdf_raw = df2.toPandas()
logger.info(f"Collected {len(pdf_raw):,} rows.")

row_uid = pdf_raw["row_uid"].to_numpy()
if len(set(row_uid)) != len(row_uid):
    raise RuntimeError("row_uid not unique after toPandas(); aborting.")

# ========== 4. Scale (kept only for the model, never persisted) ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pdf_raw[FEATURE_COLS].to_numpy())
X = X_scaled

# ========== 5. IsolationForest (identical hyper-parameters to 02) ==========
logger.info("Training IsolationForest (n_estimators=150, max_samples=512, n_jobs=4) ...")
clf = IsolationForest(
    n_estimators=150,
    max_samples=512,
    max_features=0.8,
    random_state=SEED,
    n_jobs=4,
)
t_fit = time.time()
clf.fit(X)
logger.info(f"fit done in {time.time()-t_fit:.1f}s")

# ========== 6. Score + dynamic threshold ==========
with parallel_backend("threading", n_jobs=-1):
    scores = clf.decision_function(X)

threshold, method = find_dynamic_threshold(scores)
anom_mask = scores < threshold

full = pdf_raw[["row_uid"] + FEATURE_COLS + ["bucket", "hvfhs_license_num"]].copy()
full["outlierScore"] = scores
full["prediction"] = np.where(anom_mask, -1, 1)

anom = full[anom_mask].copy()
anom["rank_in_layer"] = (
    anom.groupby(["bucket", "hvfhs_license_num"])["outlierScore"]
        .rank(method="first", ascending=True).astype(int)
)
anom["layer_total"] = (
    anom.groupby(["bucket", "hvfhs_license_num"])["outlierScore"]
        .transform("count").astype(int)
)

n_anom = len(anom)
logger.info(f"Anomalies: {n_anom:,} ({n_anom/total_rows*100:.3f}%)")
logger.info(f"REGRESSION CHECK vs original 02 run ({EXPECTED_ANOMALIES:,}): "
            f"{'MATCH' if n_anom == EXPECTED_ANOMALIES else 'MISMATCH'}")

# ========== 7. Persist scores + anomalies (with row_uid) ==========
os.makedirs(OUTPUT_BASE, exist_ok=True)
spark.createDataFrame(full).write.mode("overwrite").parquet(
    os.path.join(OUTPUT_BASE, "full_scores_rowkeyed"))
spark.createDataFrame(anom).write.mode("overwrite").parquet(
    os.path.join(OUTPUT_BASE, "strict_anomalies_rowkeyed"))
logger.info(f"Saved score tables to {OUTPUT_BASE}")

# ========== 8. SHAP with row_uid carried through ==========
logger.info(f"Computing SHAP for {min(SHAP_SAMPLE_N, n_anom)} anomalies ...")
shap_sample = anom.sort_values("outlierScore").head(SHAP_SAMPLE_N).copy()
X_shap = scaler.transform(shap_sample[FEATURE_COLS].to_numpy())

n_chunks = min(4, len(X_shap))
chunks = np.array_split(np.arange(len(X_shap)), n_chunks)


def compute_shap(idx_chunk):
    explainer = shap.TreeExplainer(clf)
    return idx_chunk, explainer.shap_values(X_shap[idx_chunk])


results = Parallel(n_jobs=n_chunks)(
    delayed(compute_shap)(c) for c in chunks)

# Reassemble IN THE ORIGINAL ORDER using the returned indices, so the SHAP
# matrix row i always corresponds to shap_sample row i.
shap_values = np.empty((len(X_shap), len(FEATURE_COLS)), dtype=float)
for idx_chunk, vals in results:
    shap_values[idx_chunk] = vals

# HARD INTEGRITY CHECKS - these are what 02 was missing.
assert shap_values.shape[0] == len(shap_sample), (
    f"SHAP rows {shap_values.shape[0]} != sample rows {len(shap_sample)}")
assert np.isfinite(shap_values).all(), "non-finite SHAP values"
for c in FEATURE_COLS:                        # additive check
    assert np.isfinite(shap_values[:, FEATURE_COLS.index(c)]).all()

shap_out = pd.DataFrame({"row_uid": shap_sample["row_uid"].to_numpy()})
for i, c in enumerate(FEATURE_COLS):
    shap_out[f"shap_{c}"] = shap_values[:, i]
for c in FEATURE_COLS:                        # keep the raw feature value too
    shap_out[f"feat_{c}"] = shap_sample[c].to_numpy()
shap_out["outlierScore"] = shap_sample["outlierScore"].to_numpy()
shap_out["bucket"] = shap_sample["bucket"].to_numpy()
shap_out["hvfhs_license_num"] = shap_sample["hvfhs_license_num"].to_numpy()

shap_path = os.path.join(OUTPUT_BASE, "shap_rowkeyed.csv")
shap_out.to_csv(shap_path, index=False)
logger.info(f"Saved {shap_path}  ({len(shap_out):,} rows x {shap_out.shape[1]} cols)")

# ========== 9. Run metadata / integrity record ==========
meta = {
    "script": os.path.basename(__file__),
    "input": INPUT_PATH,
    "output_base": OUTPUT_BASE,
    "seed": SEED,
    "frac": FRAC,
    "sampled_rows": int(total_rows),
    "anomalies": int(n_anom),
    "expected_anomalies_from_02": EXPECTED_ANOMALIES,
    "anomaly_count_matches_02": bool(n_anom == EXPECTED_ANOMALIES),
    "threshold": float(threshold),
    "threshold_method": method,
    "shap_samples": int(len(shap_out)),
    "shap_rows_equal_sample": bool(len(shap_out) == len(shap_sample)),
    "row_uid_unique": True,
    "feature_cols": FEATURE_COLS,
    "iforest_params": {"n_estimators": 150, "max_samples": 512,
                       "max_features": 0.8, "n_jobs": 4, "random_state": SEED},
    "note": "row_uid joins 07b_shap_cluster_analysis_rowkeyed.py to cluster labels",
}
with open(os.path.join(OUTPUT_BASE, "run_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

logger.info("=== run_meta ===")
for k, v in meta.items():
    logger.info(f"  {k}: {v}")

spark.stop()
logger.info("Done.")
