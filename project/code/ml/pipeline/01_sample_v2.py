# Location: project/code/ml/pipeline/01_sample_v2.py | PROJECT_ROOT depth: 3
# sample_v2.py  (now: 01_sample_v2.py; formerly project/code/ml/sample_v2.py)
# Stratified sampling script for NYC FHV data (BisectingKMeans Version).
# Reads cleaned parquet, applies feature engineering, uniform 10% stratified sampling,
# and writes sampled dataset to disk for downstream BisectingKMeans clustering.

import os
import math
import sys
import time
import logging
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, when, lit, log1p, concat_ws, sum as spark_sum,
    sin, cos, dayofweek
)

# ============ Configuration ============

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Location: ml/pipeline/01_sample_v2.py -> 3 levels up = project/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "output", "cleaned")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "output", "sampled_km")
SEED = 42

# ============ Logging (keeps a run record in project/data/log/) ============
LOG_DIR = os.path.join(PROJECT_ROOT, "data", "log")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"sample_v2_{time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
# Note: plain print() output stays on the console. The FileHandler above records
# this module's run header, and Spark's own SparkSubmit/progress logs are
# appended to the same file, so every run leaves a trace in project/data/log/.
logger.info("sample_v2 start | PROJECT_ROOT=%s", PROJECT_ROOT)
logger.info("INPUT_PATH=%s", INPUT_PATH)
logger.info("OUTPUT_PATH=%s", OUTPUT_PATH)
MIN_SAMPLES_PER_LAYER = 30

# Feature columns for BisectingKMeans
FEATURE_COLS = [
    "log_miles",
    "log_fare",
    "log_tips",
    "speed_mph",
    "log_duration",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos"
]

# Capping thresholds
MILES_CAP = 100.0
FARE_CAP = 250.0
TIPS_CAP = 100.0
SPEED_MIN = 1.0
SPEED_MAX = 80.0
DURATION_MAX = 240.0

# Uniform 10% sampling ratio for all layers
RATIO_HOT = 0.10
RATIO_WARM = 0.10
RATIO_COLD = 0.10

# ============ Spark Session ============
spark = (SparkSession.builder
    .appName("FHVSampleV2_KMeans")
    .master("local[8]")
    .config("spark.local.dir", "D:/temp_spark")
    .config("spark.sql.shuffle.partitions", "16")
    .config("spark.driver.memory", "12g")
    .config("spark.memory.fraction", "0.8")
    .config("spark.memory.storageFraction", "0.3")
    .config("spark.memory.offHeap.enabled", "true")
    .config("spark.memory.offHeap.size", "1g")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate())

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

# 3. Filter out unreasonable speeds
feat_df = feat_df.filter(
    (col("speed_mph") >= SPEED_MIN) & (col("speed_mph") <= SPEED_MAX)
)

# 4. Log-transform skewed features
feat_df = feat_df.withColumn("log_miles", log1p("trip_miles_capped"))
feat_df = feat_df.withColumn("log_fare", log1p("fare_capped"))
feat_df = feat_df.withColumn("log_tips", log1p("tips_capped"))
feat_df = feat_df.withColumn("log_duration", log1p("duration_min"))

# 5. Sin/Cos encoding for hour and day of week
# Hour: 0-23 mapped to angle [0, 2π)
feat_df = feat_df.withColumn("hour_sin", sin(col("pickup_hour") * 2.0 * math.pi / 24.0)) \
                 .withColumn("hour_cos", cos(col("pickup_hour") * 2.0 * math.pi / 24.0))

# Day of week: dayofweek returns 1(Sun)..7(Sat), subtract 1 to get 0..6, then map to [0, 2π)
feat_df = feat_df.withColumn("day_of_week_idx", dayofweek("pickup_datetime") - 1) \
                 .withColumn("day_sin", sin(col("day_of_week_idx") * 2.0 * math.pi / 7.0)) \
                 .withColumn("day_cos", cos(col("day_of_week_idx") * 2.0 * math.pi / 7.0))

feat_count = feat_df.count()
print(f"Rows after feature engineering: {feat_count:,}")

# ============ Build Stratification Key ============
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

# ============ Stratified Sampling (Uniform 10%) ============
layer_stats = feat_df.groupBy("layer_key", "bucket").count()
layer_stats.show(truncate=False)

def compute_sample_fraction(count, bucket):
    if bucket == "hot":
        ratio = RATIO_HOT
    elif bucket == "warm":
        ratio = RATIO_WARM
    else: # cold
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
    "pickup_month"
]
output_df = sampled_df.select(*persist_cols)

# ============ Write Output ============
os.makedirs(OUTPUT_PATH, exist_ok=True)
output_df.write.mode("overwrite").parquet(OUTPUT_PATH)
print(f"Sampled data written to {OUTPUT_PATH}")

# ============ Quality Report ============
print("\n===== Sampling Quality Report (v2 KMeans) =====")
for col_name in FEATURE_COLS:
    orig_mean = feat_df.agg({col_name: "mean"}).collect()[0][0]
    samp_mean = output_df.agg({col_name: "mean"}).collect()[0][0]
    bias = abs(orig_mean - samp_mean) / abs(orig_mean) if orig_mean != 0 else 0
    print(f"{col_name}: original mean={orig_mean:.4f}, sample mean={samp_mean:.4f}, bias={bias:.2%}")

# Layer distribution check
print("Layer distribution:")
output_df.groupBy("bucket").count().show()

# Company proportion check
for lic in ["HV0003", "HV0005"]:
    orig_pct = feat_df.filter(col("hvfhs_license_num") == lic).count() / feat_count
    samp_pct = output_df.filter(col("hvfhs_license_num") == lic).count() / final_count
    print(f"{lic}: original={orig_pct:.2%}, sample={samp_pct:.2%}, diff={abs(orig_pct-samp_pct):.2%}")

# Long-trip retention (>50 miles)
LOG50 = math.log1p(50)
long_orig = feat_df.filter(col("trip_miles_capped") > 50).count() / feat_count
long_samp = output_df.filter(col("log_miles") > LOG50).count() / final_count
print(f"Long trips (>50mi): original={long_orig:.4%}, sample={long_samp:.4%}")

print("Sampling complete.")
spark.stop()