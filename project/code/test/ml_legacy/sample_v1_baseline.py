# Location: project/code/test/ml_legacy/sample_v1_baseline.py | PROJECT_ROOT depth: 3
# 01_sample.py
# Stratified sampling script for NYC FHV data.
# Reads cleaned parquet, applies feature engineering, stratified sampling,
# and writes sampled dataset to disk for downstream tuning.

import os
import sys
import math
import time
import logging
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, when, lit, log1p, concat_ws, sum as spark_sum
)

# ============ Configuration ============

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Location: code/test/ml_legacy/sample_v1_baseline.py -> 4 levels up = project/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))  # .../project
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "output", "cleaned")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "output", "sampled")
SEED = 42

# ============ Logging (run record in project/data/log/) ============
LOG_DIR = os.path.join(PROJECT_ROOT, "data", "log")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "sample_v1_baseline_" + time.strftime("%Y%m%d_%H%M%S") + ".log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.info("start | PROJECT_ROOT=%s | INPUT_PATH=%s | OUTPUT_PATH=%s", PROJECT_ROOT, INPUT_PATH, OUTPUT_PATH)
MIN_SAMPLES_PER_LAYER = 30

# Feature columns (computed from cleaned fields)
FEATURE_COLS = [
    "log_miles",
    "log_fare",
    "log_tips",
    "speed_mph",
    "log_duration"
]

# Capping thresholds (based on EDA, applied after reading cleaned data)
MILES_CAP = 100.0
FARE_CAP = 250.0
TIPS_CAP = 100.0
SPEED_MIN = 1.0
SPEED_MAX = 80.0          # relaxed from 85 to 80 to keep more normal trips
DURATION_MAX = 240.0      # safety margin; trip_valid already ensures <=180

# Sampling ratios per bucket
RATIO_HOT = 0.08
RATIO_WARM = 0.20
RATIO_COLD = 1.0

# ============ Spark Session ============
spark = SparkSession.builder \
    .appName("FHVSample") \
    .config("spark.local.dir", "D:/temp_spark") \
    .config("spark.sql.shuffle.partitions", "400") \
    .config("spark.driver.memory", "16g") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "3g") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()

# ============ Read Cleaned Data ============
df = spark.read.parquet(INPUT_PATH)
total_raw = df.count()
print(f"Total raw rows (cleaned): {total_raw:,}")

# ============ Filter using validation flags ============
valid_df = df.filter(
    col("fare_nonneg") == 1
).filter(
    col("trip_valid") == 1
).filter(
    col("wait_valid") == 1
)
# Cache removed to avoid OOM – let Spark recompute from source when needed
valid_count = valid_df.count()
print(f"Rows after validation filters: {valid_count:,} ({valid_count/total_raw*100:.1f}%)")

# ============ Feature Engineering ============
# 1. Cap extreme values
feat_df = valid_df.withColumn(
    "trip_miles_capped",
    when(col("trip_miles") > MILES_CAP, MILES_CAP).otherwise(col("trip_miles"))
).withColumn(
    "fare_capped",
    when(col("base_passenger_fare") > FARE_CAP, FARE_CAP).otherwise(col("base_passenger_fare"))
).withColumn(
    "tips_capped",
    when(col("tips") > TIPS_CAP, TIPS_CAP).otherwise(col("tips"))
).withColumn(
    "duration_min",
    col("trip_duration_minutes")
)

# 2. Compute speed_mph
feat_df = feat_df.withColumn(
    "speed_mph",
    when(col("duration_min") > 0,
         col("trip_miles_capped") / (col("duration_min") / 60.0)
    ).otherwise(None)
)

# 3. Filter out unreasonable speeds (GPS noise or very slow traffic)
feat_df = feat_df.filter(
    (col("speed_mph") >= SPEED_MIN) & (col("speed_mph") <= SPEED_MAX)
)

# 4. Log-transform skewed features
feat_df = feat_df.withColumn("log_miles", log1p("trip_miles_capped"))
feat_df = feat_df.withColumn("log_fare", log1p("fare_capped"))
feat_df = feat_df.withColumn("log_tips", log1p("tips_capped"))
feat_df = feat_df.withColumn("log_duration", log1p("duration_min"))

# Cache removed here as well
feat_count = feat_df.count()
print(f"Rows after speed filter and feature engineering: {feat_count:,}")

# ============ Build Stratification Key ============
# Per-company cumulative percentage to assign hot/warm/cold buckets
window_spec = Window.partitionBy("hvfhs_license_num") \
                     .orderBy(col("count").desc()) \
                     .rowsBetween(Window.unboundedPreceding, Window.currentRow)

area_counts = feat_df.groupBy("PULocationID", "hvfhs_license_num").count()
total_per_company = feat_df.groupBy("hvfhs_license_num").count() \
                           .withColumnRenamed("count", "total_company")

area_counts = area_counts.join(total_per_company, "hvfhs_license_num") \
    .withColumn("cum_pct", spark_sum(col("count")).over(window_spec) / col("total_company"))

area_bucket = area_counts.withColumn(
    "bucket",
    when(col("cum_pct") <= 0.75, lit("hot"))
    .when(col("cum_pct") <= 0.92, lit("warm"))
    .otherwise(lit("cold"))
).select("PULocationID", "hvfhs_license_num", "bucket")

feat_df = feat_df.join(area_bucket, on=["PULocationID", "hvfhs_license_num"], how="left")
feat_df = feat_df.withColumn("layer_key",
    concat_ws("_", col("bucket"), col("hvfhs_license_num"))
)

# ============ Stratified Sampling ============
layer_stats = feat_df.groupBy("layer_key", "bucket").count()
layer_stats.show(truncate=False)

def compute_sample_fraction(count, bucket):
    if bucket == "hot":
        ratio = RATIO_HOT
    elif bucket == "warm":
        ratio = RATIO_WARM
    else:   # cold
        ratio = RATIO_COLD
    desired = max(int(count * ratio), MIN_SAMPLES_PER_LAYER)
    fraction = min(desired / count, 1.0)
    return fraction

fractions = {}
for row in layer_stats.collect():
    layer_key = row.layer_key
    bucket = row.bucket
    count = row["count"]
    fraction = compute_sample_fraction(count, bucket)
    fractions[layer_key] = fraction

sampled_df = feat_df.sampleBy("layer_key", fractions, seed=SEED)
final_count = sampled_df.count()
print(f"Sampled dataset size: {final_count:,} (overall rate: {final_count/feat_count*100:.1f}%)")

# ============ Select Columns to Persist ============
persist_cols = FEATURE_COLS + [
    "PULocationID",
    "hvfhs_license_num",
    "bucket",
    "layer_key",
    "pickup_year",
    "pickup_month",
    "pickup_hour",
    "pickup_day_of_week"
]
output_df = sampled_df.select(*persist_cols)

# ============ Write Output ============
os.makedirs(OUTPUT_PATH, exist_ok=True)
output_df.write.mode("overwrite").parquet(OUTPUT_PATH)
print(f"Sampled data written to {OUTPUT_PATH}")

# ============ Quick Quality Report ============
print("\n===== Sampling Quality Report =====")
for col_name in FEATURE_COLS:
    orig_mean = feat_df.agg({col_name: "mean"}).collect()[0][0]
    samp_mean = output_df.agg({col_name: "mean"}).collect()[0][0]
    bias = abs(orig_mean - samp_mean) / abs(orig_mean) if orig_mean != 0 else 0
    print(f"{col_name}: original mean={orig_mean:.4f}, sample mean={samp_mean:.4f}, bias={bias:.2%}")

# Company proportion check
for lic in ["HV0003", "HV0005"]:
    orig_pct = feat_df.filter(col("hvfhs_license_num") == lic).count() / feat_count
    samp_pct = output_df.filter(col("hvfhs_license_num") == lic).count() / final_count
    print(f"{lic}: original={orig_pct:.2%}, sample={samp_pct:.2%}, diff={abs(orig_pct-samp_pct):.2%}")

# Long-trip retention (>50 miles)
LOG50 = math.log1p(50)   # ≈ 3.9318
long_orig = feat_df.filter(col("trip_miles_capped") > 50).count() / feat_count
long_samp = output_df.filter(col("log_miles") > LOG50).count() / final_count
print(f"Long trips (>50mi): original={long_orig:.4%}, sample={long_samp:.4%}")

# Cold layer coverage
cold_orig = feat_df.filter(col("bucket") == "cold").count()
cold_samp = output_df.filter(col("bucket") == "cold").count()
if cold_orig > 0:
    print(f"Cold layers: original={cold_orig:,}, sample={cold_samp:,}, retention={cold_samp/cold_orig:.2%}")
else:
    print("No cold layers found.")

print("Sampling complete.")
spark.stop()