"""
bi_aggregation.py - Generate two-level aggregated tables for BI dashboards.
Reads cleaned Parquet from ../data/output/cleaned, outputs L1 and L2 aggregations.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

CLEANED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "output", "cleaned")
BI_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output", "bi_aggregates")

os.makedirs(BI_OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Spark Session (same config as preprocessing)
# ------------------------------------------------------------
spark = SparkSession.builder \
    .appName("FHVHV_BI_Aggregation") \
    .master("local[*]") \
    .config("spark.driver.memory", "12g") \
    .config("spark.default.parallelism", "40") \
    .config("spark.sql.shuffle.partitions", "40") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.local.dir", "D:/temp_spark") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print(f"[INFO] Reading cleaned data from: {CLEANED_DATA_DIR}")

# ------------------------------------------------------------
# Step 1: Load cleaned data
# ------------------------------------------------------------
df = spark.read.parquet(CLEANED_DATA_DIR)
total_rows = df.count()
print(f"[INFO] Total rows loaded: {total_rows:,}")

# ------------------------------------------------------------
# Step 2: Prepare date column for grouping
# ------------------------------------------------------------
df_agg = df.withColumn("pickup_date", F.to_date("pickup_datetime"))

# ------------------------------------------------------------
# Step 3: L1 aggregation – date × hour × PULocationID × company
# ------------------------------------------------------------
# Key changes:
#   1) avg_wait_minutes now only averages rows where wait_valid = 1
#      (i.e. wait_minutes between 0 and 120). Rows with wait_valid = 0
#      are excluded via F.when(...), which produces NULL for those rows,
#      and Spark's avg() ignores NULLs.
#   2) New column total_valid_wait_minutes: sum of wait_minutes for
#      valid rows only, giving the total waiting burden per group.
l1 = df_agg.groupBy(
    "pickup_date", "pickup_hour", "PULocationID", "hvfhs_license_num"
).agg(
    F.count("*").alias("trip_count"),
    F.sum("base_passenger_fare").alias("total_fare"),
    # Average wait time computed ONLY on valid waits (wait_valid = 1)
    F.avg(
        F.when(F.col("wait_valid") == 1, F.col("wait_minutes"))
    ).alias("avg_wait_minutes"),
    # Total valid wait minutes (sum of wait_minutes where wait_valid = 1)
    F.sum(
        F.when(F.col("wait_valid") == 1, F.col("wait_minutes"))
    ).alias("total_valid_wait_minutes"),
    F.avg("trip_duration_minutes").alias("avg_trip_duration"),
    F.sum("trip_miles").alias("total_miles"),
    F.sum("tips").alias("total_tips"),
    F.sum(F.col("wait_valid")).alias("valid_wait_count"),
    F.sum(F.col("fare_nonneg")).alias("valid_fare_count"),
    F.sum(F.col("trip_valid")).alias("valid_trip_count")
)

# ------------------------------------------------------------
# Step 4: L2 aggregation – date × hour × PULocationID (no company)
# ------------------------------------------------------------
# Same fix applied to L2 so the overview table stays consistent
# with the L1 detail table.
l2 = df_agg.groupBy(
    "pickup_date", "pickup_hour", "PULocationID"
).agg(
    F.count("*").alias("trip_count"),
    F.sum("base_passenger_fare").alias("total_fare"),
    # Average wait time computed ONLY on valid waits (wait_valid = 1)
    F.avg(
        F.when(F.col("wait_valid") == 1, F.col("wait_minutes"))
    ).alias("avg_wait_minutes"),
    # Total valid wait minutes (sum of wait_minutes where wait_valid = 1)
    F.sum(
        F.when(F.col("wait_valid") == 1, F.col("wait_minutes"))
    ).alias("total_valid_wait_minutes"),
    F.avg("trip_duration_minutes").alias("avg_trip_duration"),
    F.sum("trip_miles").alias("total_miles"),
    F.sum("tips").alias("total_tips")
)

# ------------------------------------------------------------
# Step 5: Write both tables
# ------------------------------------------------------------
l1_path = os.path.join(BI_OUTPUT_DIR, "l1_detail")
l2_path = os.path.join(BI_OUTPUT_DIR, "l2_overview")

print(f"[INFO] Writing L1 (detail) to: {l1_path}")
l1.write.mode("overwrite").partitionBy("pickup_date").parquet(l1_path)

print(f"[INFO] Writing L2 (overview) to: {l2_path}")
l2.write.mode("overwrite").partitionBy("pickup_date").parquet(l2_path)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
l1_count = l1.count()
l2_count = l2.count()
print("\n========== BI Aggregation Summary ==========")
print(f"Cleaned input rows: {total_rows:,}")
print(f"L1 detail rows:     {l1_count:,}")
print(f"L2 overview rows:   {l2_count:,}")
print(f"Output location:    {BI_OUTPUT_DIR}")
print("============================================")

spark.stop()
