"""
Location: project/code/ml/pipeline/02_isolation_forest_shap.py | PROJECT_ROOT depth: 3
(formerly project/code/ml/isolation-forest-SHAP.py)

Taxi Trip Anomaly Detection using Isolation Forest with Hybrid Parallelism
- Runs on 10% of the 10% sample (i.e., 1% of original data, ~4.67M rows)
- Dynamic threshold: elbow method + IQR fallback
"""

import os, sys, time, logging
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend, Parallel, delayed
import shap

# ========== Logging Setup ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Location: ml/pipeline/02_isolation_forest_shap.py -> 3 levels up = project/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
LOG_DIR = os.path.join(PROJECT_ROOT, "data", "log")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"if_10pct_of_10pct_{time.strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ========== Configuration ==========
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "output", "sampled_km")  # Full 10% sample
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "data", "output", "if_results_10pct_of_10pct")
FEATURE_COLS = [
    "log_miles", "log_fare", "log_tips", "speed_mph", "log_duration",
    "hour_sin", "hour_cos", "day_sin", "day_cos"
]
SEED = 42
FRAC = 0.10                     # 10% of the 10% sample → ~4.67M rows

# ========== Dynamic Threshold Function ==========
def find_dynamic_threshold(scores, min_anomaly_ratio=0.001, max_anomaly_ratio=0.05):
    sorted_scores = np.sort(scores)
    diffs = np.diff(sorted_scores)
    elbow_idx = np.argmax(diffs)
    elbow_threshold = sorted_scores[elbow_idx]
    elbow_ratio = (scores < elbow_threshold).sum() / len(scores)

    if min_anomaly_ratio <= elbow_ratio <= max_anomaly_ratio:
        logger.info(f"Elbow method: threshold={elbow_threshold:.4f}, anomaly ratio={elbow_ratio*100:.2f}%")
        return elbow_threshold, "elbow"

    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    iqr_threshold = q1 - 3.0 * iqr
    iqr_ratio = (scores < iqr_threshold).sum() / len(scores)
    logger.info(f"IQR fallback: threshold={iqr_threshold:.4f}, anomaly ratio={iqr_ratio*100:.2f}%")
    return iqr_threshold, "iqr"

# ========== 1. Spark Session ==========
spark = (SparkSession.builder
         .appName("IF_10pctOf10pct")
         .master("local[6]")
         .config("spark.driver.memory", "8g")
         .config("spark.driver.maxResultSize", "2g")
         .config("spark.sql.shuffle.partitions", "24")
         .config("spark.local.dir", "D:/temp_spark")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
logger.info("Spark session started (distributed loading & sampling).")

# ========== 2. Distributed Read & Stratified Sampling ==========
logger.info("[PARALLEL] Reading 10% sample parquet with Spark...")
df10 = spark.read.parquet(INPUT_PATH)

logger.info("[PARALLEL] Stratified sampling per bucket (distributed sampleBy)...")
bucket_keys = [row.bucket for row in df10.select("bucket").distinct().collect()]
fracs = {k: FRAC for k in bucket_keys}
df2 = df10.sampleBy("bucket", fractions=fracs, seed=SEED) \
           .select(FEATURE_COLS + ["bucket", "hvfhs_license_num"])
df2.cache()
total_rows = df2.count()
logger.info(f"Sampled 10% of 10% sample => {total_rows:,} rows (distributed sampling done).")

# ========== 3. Collect to Pandas ==========
logger.info("Collecting sampled features to driver...")
pdf_raw = df2.toPandas()
logger.info(f"Collected {len(pdf_raw):,} rows into pandas DataFrame.")

# ========== 4. Feature Scaling via sklearn ==========
logger.info("[PARALLEL] StandardScaler using sklearn (vectorized, multi-threaded BLAS)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pdf_raw[FEATURE_COLS].values)
pdf = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
pdf[["bucket", "hvfhs_license_num"]] = pdf_raw[["bucket", "hvfhs_license_num"]]
logger.info("Feature scaling completed.")

X = pdf[FEATURE_COLS].values

# ========== 5. Train IsolationForest (reduced parallelism for memory safety) ==========
logger.info("[PARALLEL] Training IsolationForest (n_jobs=4, max_samples=512)...")
clf = IsolationForest(
    n_estimators=150,
    max_samples=512,
    max_features=0.8,
    random_state=SEED,
    n_jobs=4                    # Use 4 processes instead of -1 to avoid memory overload
)
clf.fit(X)
logger.info("Training completed (parallel tree construction).")

# ========== 6. Compute scores and apply dynamic threshold ==========
logger.info("[PARALLEL] Computing anomaly scores (parallel_backend threading)...")
with parallel_backend("threading", n_jobs=-1):
    scores = clf.decision_function(X)

threshold, method = find_dynamic_threshold(scores)
anom_mask = scores < threshold

pdf["outlierScore"] = scores
pdf["prediction"] = np.where(anom_mask, -1, 1)

anom_pdf = pdf[anom_mask].copy()
logger.info(f"Anomalies detected (dynamic threshold, method={method}): {len(anom_pdf):,} ({len(anom_pdf)/total_rows*100:.2f}%)")

# Per-layer ranking
if len(anom_pdf) > 0:
    anom_pdf["rank_in_layer"] = anom_pdf.groupby(["bucket", "hvfhs_license_num"])["outlierScore"].rank(
        method="first", ascending=True
    ).astype(int)
    anom_pdf["layer_total"] = anom_pdf.groupby(["bucket", "hvfhs_license_num"])["outlierScore"].transform("count")

# ========== 7. Save results ==========
os.makedirs(OUTPUT_BASE, exist_ok=True)
pdf[FEATURE_COLS] = pdf_raw[FEATURE_COLS]
anom_pdf[FEATURE_COLS] = pdf_raw.loc[pdf_raw.index.isin(anom_pdf.index), FEATURE_COLS].values

pdf.to_parquet(os.path.join(OUTPUT_BASE, "full_scores"), index=False)
anom_pdf.to_parquet(os.path.join(OUTPUT_BASE, "strict_anomalies"), index=False)
logger.info(f"Results saved to {OUTPUT_BASE}")

# ========== 8. SHAP explanation (multi-process parallel, 2000 samples) ==========
if len(anom_pdf) > 0:
    logger.info("[PARALLEL] Computing SHAP values (multi-process over chunks, 2000 samples)...")
    shap_sample = anom_pdf.head(2000)
    X_shap = shap_sample[FEATURE_COLS].values

    n_chunks = min(4, len(X_shap))
    chunks = np.array_split(X_shap, n_chunks)

    def compute_shap(chunk):
        explainer = shap.TreeExplainer(clf)
        return explainer.shap_values(chunk)

    shap_results = Parallel(n_jobs=n_chunks)(delayed(compute_shap)(chunk) for chunk in chunks)
    shap_values = np.vstack(shap_results)

    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in FEATURE_COLS])
    shap_df.to_csv(os.path.join(OUTPUT_BASE, "shap_strict_anomalies.csv"), index=False)
    logger.info("SHAP values saved (parallel computation, 2000 samples).")
else:
    logger.info("No anomalies found; SHAP skipped.")

# ========== Summary ==========
logger.info(f"Total sampled rows: {total_rows:,}")
logger.info(f"Anomalies selected: {len(anom_pdf):,}")
logger.info(f"Anomaly ratio: {len(anom_pdf)/total_rows*100:.3f}%")
logger.info("Script finished successfully.")

# ========== Simple Quality Validation ==========
logger.info("=== Quick Quality Check ===")

group_mean = pdf.groupby("prediction")[FEATURE_COLS].mean()
diff = group_mean.loc[-1] - group_mean.loc[1]
top_features = diff.abs().sort_values(ascending=False).head(5)
logger.info("Features with largest mean difference between anomaly and normal:")
for feat, val in top_features.items():
    direction = "higher" if val > 0 else "lower"
    logger.info(f"  {feat}: anomaly mean is {direction} by {abs(val):.3f}")

top10 = pdf.sort_values("outlierScore").head(10)[
    FEATURE_COLS + ["outlierScore", "bucket", "hvfhs_license_num"]
]
logger.info("Top 10 most anomalous samples:")
for idx, row in top10.iterrows():
    logger.info(f"  Score={row['outlierScore']:.4f} | bucket={row['bucket']} | "
                f"fare={row['log_fare']:.2f}, miles={row['log_miles']:.2f}, "
                f"speed={row['speed_mph']:.2f}, duration={row['log_duration']:.2f}")

bucket_rate = pdf.groupby("bucket").apply(lambda g: (g["prediction"] == -1).mean())
logger.info("Anomaly rate per bucket:")
for bucket, rate in bucket_rate.items():
    logger.info(f"  {bucket}: {rate*100:.2f}%")

logger.info("Quick validation complete.")

spark.stop()