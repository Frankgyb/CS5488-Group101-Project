"""
preprocessing.py - Clean and feature-engineer FHVHV trip records
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ------------------------------------------------------------
# Configuration – absolute paths based on script location
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "parquet")          # input Parquet folder
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output", "cleaned")  # output folder
LOG_DIR = os.path.join(PROJECT_ROOT, "data", "log")               # optional logs

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ------------------------------------------------------------
# Spark session – same config as your working EDA script
# ------------------------------------------------------------
spark = SparkSession.builder \
    .appName("FHVHV_Preprocessing") \
    .master("local[*]") \
    .config("spark.driver.memory", "12g") \
    .config("spark.default.parallelism", "40") \
    .config("spark.sql.shuffle.partitions", "40") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.local.dir", "D:/temp_spark") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print(f"[INFO] Reading Parquet data from: {DATA_DIR}")

# ------------------------------------------------------------
# Step 1: Read all Parquet files
# ------------------------------------------------------------
raw_df = spark.read.parquet(DATA_DIR)
raw_df.printSchema()
total_rows = raw_df.count()
print(f"[INFO] Total rows loaded: {total_rows:,}")

# ------------------------------------------------------------
# Step 2: Derive essential features using only existing columns
# ------------------------------------------------------------
df = raw_df \
    .withColumn("wait_minutes",
                (F.unix_timestamp("pickup_datetime") -
                 F.unix_timestamp("request_datetime")) / 60.0) \
    .withColumn("trip_duration_minutes",
                (F.unix_timestamp("dropoff_datetime") -
                 F.unix_timestamp("pickup_datetime")) / 60.0) \
    .withColumn("pickup_hour", F.hour("pickup_datetime")) \
    .withColumn("pickup_day_of_week", F.dayofweek("pickup_datetime")) \
    .withColumn("pickup_year", F.year("pickup_datetime")) \
    .withColumn("pickup_month", F.month("pickup_datetime")) \
    .withColumn("is_shared_request",
                F.when(F.col("shared_request_flag") == "Y", 1).otherwise(0)) \
    .withColumn("is_shared_match",
                F.when(F.col("shared_match_flag") == "Y", 1).otherwise(0)) \
    .withColumn("is_wav_request",
                F.when(F.col("wav_request_flag") == "Y", 1).otherwise(0)) \
    .withColumn("is_wav_match",
                F.when(F.col("wav_match_flag") == "Y", 1).otherwise(0))

# ------------------------------------------------------------
# Step 3: Add validity flags (keep all rows, flag anomalies)
# ------------------------------------------------------------
df = df \
    .withColumn("wait_valid",
                F.when((F.col("wait_minutes") >= 0) & (F.col("wait_minutes") <= 120), 1).otherwise(0)) \
    .withColumn("fare_nonneg",
                F.when(F.col("base_passenger_fare") >= 0, 1).otherwise(0)) \
    .withColumn("trip_valid",
                F.when((F.col("trip_duration_minutes") >= 1) &
                       (F.col("trip_duration_minutes") <= 180) &
                       (F.col("trip_miles") > 0), 1).otherwise(0))

# ------------------------------------------------------------
# Step 4: Write the processed data partitioned by year/month
# ------------------------------------------------------------
print(f"[INFO] Writing cleaned data to: {OUTPUT_DIR}")
df.write.mode("overwrite") \
  .partitionBy("pickup_year", "pickup_month") \
  .parquet(OUTPUT_DIR)

# ------------------------------------------------------------
# Summary report
# ------------------------------------------------------------
valid_counts = df.agg(
    F.sum("wait_valid").alias("wait_valid_count"),
    F.sum("fare_nonneg").alias("fare_nonneg_count"),
    F.sum("trip_valid").alias("trip_valid_count")
).collect()[0]

print("\n========== Processing Summary ==========")
print(f"Total rows: {total_rows:,}")
print(f"Wait time valid (0–120 min): {valid_counts['wait_valid_count']:,}")
print(f"Non-negative fare:           {valid_counts['fare_nonneg_count']:,}")
print(f"Valid trip (duration>0, miles>0): {valid_counts['trip_valid_count']:,}")
print(f"Output location: {OUTPUT_DIR}")
print("========================================")

spark.stop()