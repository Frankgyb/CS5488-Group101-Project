import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType, TimestampType

# -------------------- Configuration --------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "parquet", "*.parquet")
LOG_DIR = os.path.join(PROJECT_ROOT, "data", "log")

os.environ['HADOOP_HOME'] = r"D:\hadoop"
os.environ['PATH'] = os.environ['HADOOP_HOME'] + r"\bin;" + os.environ['PATH']

# -------------------- Spark Session --------------------
spark = SparkSession.builder \
    .appName("FHVHV_Full_EDA") \
    .master("local[*]") \
    .config("spark.driver.memory", "12g") \
    .config("spark.default.parallelism", "40") \
    .config("spark.sql.shuffle.partitions", "40") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.local.dir", "D:/temp_spark") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# -------------------- Alias mapping --------------------
COLUMN_ALIAS = {
    "hvfhs_license_num": "License Number (hvfhs_license_num)",
    "dispatching_base_num": "Base Number (dispatching_base_num)",
    "originating_base_num": "Originating Base (originating_base_num)",
    "request_datetime": "Request Time (request_datetime)",
    "on_scene_datetime": "On-Scene Time (on_scene_datetime)",
    "pickup_datetime": "Pickup Time (pickup_datetime)",
    "dropoff_datetime": "Dropoff Time (dropoff_datetime)",
    "PULocationID": "Pickup Location ID (PULocationID)",
    "DOLocationID": "Dropoff Location ID (DOLocationID)",
    "trip_miles": "Trip Miles (trip_miles)",
    "trip_time": "Trip Time (seconds) (trip_time)",
    "base_passenger_fare": "Base Fare (base_passenger_fare)",
    "tolls": "Tolls (tolls)",
    "bcf": "BCF (bcf)",
    "sales_tax": "Sales Tax (sales_tax)",
    "congestion_surcharge": "Congestion Surcharge (congestion_surcharge)",
    "airport_fee": "Airport Fee (airport_fee)",
    "tips": "Tips (tips)",
    "driver_pay": "Driver Pay (driver_pay)",
    "shared_request_flag": "Shared Request Flag (shared_request_flag)",
    "shared_match_flag": "Shared Match Flag (shared_match_flag)",
    "access_a_ride_flag": "Access-a-Ride Flag (access_a_ride_flag)",
    "wav_request_flag": "WAV Request Flag (wav_request_flag)",
    "wav_match_flag": "WAV Match Flag (wav_match_flag)",
    "wait_minutes": "Wait Time (minutes) (wait_minutes)"
}

# -------------------- Logger --------------------
log_lines = []
def log(msg):
    log_lines.append(str(msg))
    print(msg)

# -------------------- Helper functions --------------------
def numeric_stats(df, col_name, alias):
    """Compute full numeric statistics including percentiles."""
    # Basic stats
    stats_row = df.agg(
        F.count("*").alias("count"),
        F.sum(F.when(F.col(col_name).isNull(), 1).otherwise(0)).alias("null_count"),
        F.mean(col_name).alias("mean"),
        F.stddev(col_name).alias("stddev"),
        F.min(col_name).alias("min"),
        F.max(col_name).alias("max")
    ).collect()[0]

    # Approx percentiles
    percentiles = df.approxQuantile(col_name, [0.5, 0.9, 0.95, 0.99], 0.01)

    log(f"[Field: {alias}]")
    log(f"  Type: numeric")
    log(f"  Count: {stats_row['count']}")
    log(f"  Null Count: {stats_row['null_count']}")
    log(f"  Mean: {stats_row['mean']:.4f}")
    log(f"  StdDev: {stats_row['stddev']:.4f}")
    log(f"  Min: {stats_row['min']:.4f}")
    log(f"  Max: {stats_row['max']:.4f}")
    log(f"  Percentiles: P50={percentiles[0]:.4f}, P90={percentiles[1]:.4f}, "
        f"P95={percentiles[2]:.4f}, P99={percentiles[3]:.4f}")
    log("")

def categorical_stats(df, col_name, alias, top_n=20):
    """Compute frequency distribution for categorical/string fields."""
    total = df.count()
    null_count = df.filter(F.col(col_name).isNull()).count()
    distinct_count = df.select(F.approx_count_distinct(col_name)).collect()[0][0]

    # Top-N values
    top_values = df.groupBy(col_name).agg(F.count("*").alias("cnt")) \
                   .orderBy(F.desc("cnt")).limit(top_n).collect()

    log(f"[Field: {alias}]")
    log(f"  Type: string")
    log(f"  Total Count: {total}")
    log(f"  Null Count: {null_count}")
    log(f"  Distinct Values (approx): {distinct_count}")
    log(f"  Top {top_n} Values:")
    for row in top_values:
        val = row[col_name] if row[col_name] is not None else "NULL"
        pct = row['cnt'] / total * 100
        log(f"    {val}: {row['cnt']} ({pct:.2f}%)")
    log("")

def timestamp_stats(df, col_name, alias):
    """Compute temporal range and monthly distribution for timestamp fields."""
    total = df.count()
    null_count = df.filter(F.col(col_name).isNull()).count()

    # Range
    range_row = df.agg(
        F.min(col_name).alias("min_ts"),
        F.max(col_name).alias("max_ts")
    ).collect()[0]

    # Monthly distribution
    month_col = F.date_format(col_name, "yyyy-MM")
    monthly = df.groupBy(month_col.alias("month")).agg(F.count("*").alias("cnt")) \
                .orderBy("month").collect()

    log(f"[Field: {alias}]")
    log(f"  Type: timestamp")
    log(f"  Total Count: {total}")
    log(f"  Null Count: {null_count}")
    log(f"  Earliest: {range_row['min_ts']}")
    log(f"  Latest: {range_row['max_ts']}")
    log(f"  Monthly Distribution (first 24 months):")
    for row in monthly[:24]:
        log(f"    {row['month']}: {row['cnt']}")
    if len(monthly) > 24:
        log(f"    ... (total {len(monthly)} months)")
    log("")

# -------------------- Main Analysis --------------------
# Load data and derive wait_minutes
raw_df = spark.read.parquet(INPUT_PATH)
df = raw_df.withColumn(
    "wait_minutes",
    (F.unix_timestamp("pickup_datetime") - F.unix_timestamp("request_datetime")) / 60.0
)

# Get all column names (including derived wait_minutes)
all_columns = df.columns  # list of original columns + wait_minutes

# Iterate over each column and compute appropriate stats
for col_name in all_columns:
    alias = COLUMN_ALIAS.get(col_name, col_name)  # fallback to original name if missing
    dtype = df.schema[col_name].dataType

    if isinstance(dtype, NumericType):
        numeric_stats(df, col_name, alias)
    elif isinstance(dtype, StringType):
        categorical_stats(df, col_name, alias)
    elif isinstance(dtype, TimestampType):
        timestamp_stats(df, col_name, alias)
    else:
        # Fallback: basic info
        log(f"[Field: {alias}]")
        log(f"  Type: {dtype}")
        log(f"  (No detailed stats implemented for this type)")
        log("")

# -------------------- Save log --------------------
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"full_eda_log_{timestamp}.txt")
with open(log_file, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

log(f"[Done] Full EDA log saved to: {log_file}")

spark.stop()