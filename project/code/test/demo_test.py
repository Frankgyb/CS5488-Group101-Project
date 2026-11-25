import os
import time
import sys
from pathlib import Path

# Location: project/code/test/demo_test.py | PROJECT_ROOT depth: 2
PROJECT_ROOT = Path(__file__).resolve().parents[2]      # .../project

# --- Windows Spark Environment Fix ---
# Ensure Python worker uses the same Python interpreter as PyCharm
os.environ['PYSPARK_PYTHON'] = sys.executable if 'sys' in globals() else 'python'
# If you have configured HADOOP_HOME in system variables, you don't need to set it here.
# os.environ['HADOOP_HOME'] = r"D:\hadoop"

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType


def main():
    print("=" * 60)
    print("Starting PySpark Demo - Single Month Data")
    print("=" * 60)

    # 1. Initialize Spark Session (Use local[4] for demo to avoid heating up your CPU)
    spark = SparkSession.builder \
        .appName("CS5488-Demo-Test") \
        .master("local[4]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # Set log level to WARN to avoid massive INFO logs in PyCharm
    spark.sparkContext.setLogLevel("WARN")
    print("[INFO] Spark Session created successfully.")

    # 2. Define the file path (resolved from PROJECT_ROOT, no absolute drive letters)
    file_path = PROJECT_ROOT / "data" / "parquet" / "fhvhv_tripdata_2023-01.parquet"

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return

    # 3. Read Parquet
    print(f"\n[INFO] Reading Parquet file: {file_path}")
    t0 = time.time()
    df = spark.read.parquet(file_path)

    # Print Schema to verify columns
    print("\n[INFO] DataFrame Schema:")
    df.printSchema()

    # 4. Basic Action: Count rows
    print("\n[INFO] Counting total rows...")
    total_rows = df.count()
    print(f"[RESULT] Total rows in Jan 2023: {total_rows:,}")

    # 5. Data Cleaning & Feature Engineering (Based on handover doc)
    print("\n[INFO] Applying cleaning rules...")

    # Rule 1: Drop null request or pickup datetimes
    df_clean = df.dropna(subset=["request_datetime", "pickup_datetime"])

    # Rule 2: Calculate wait_seconds and filter anomalies
    # request_datetime and pickup_datetime are usually timestamp types in Parquet
    df_clean = df_clean.withColumn(
        "wait_seconds",
        F.unix_timestamp("pickup_datetime") - F.unix_timestamp("request_datetime")
    )

    # Filter: wait_seconds >= 0 and <= 86400 (24 hours)
    df_clean = df_clean.filter((F.col("wait_seconds") >= 0) & (F.col("wait_seconds") <= 86400))

    # Rule 3: Drop null or invalid PULocationID (usually IDs > 263 are invalid)
    df_clean = df_clean.filter(
        F.col("PULocationID").isNotNull() &
        (F.col("PULocationID") <= 263) &
        (F.col("PULocationID") > 0)
    )

    # 6. Basic Aggregation (Wait time statistics)
    print("\n[INFO] Calculating Wait Time Statistics...")
    # Average and max wait time
    stats = df_clean.agg(
        F.count("*").alias("valid_rows"),
        F.round(F.avg("wait_seconds"), 2).alias("avg_wait_sec"),
        F.max("wait_seconds").alias("max_wait_sec")
    ).collect()[0]

    print("\n--- Wait Time Statistics ---")
    print(f"Valid Rows: {stats['valid_rows']:,}")
    print(f"Avg Wait Time: {stats['avg_wait_sec']} seconds")
    print(f"Max Wait Time: {stats['max_wait_sec']} seconds")

    # 7. GroupBy PULocationID (Similar to Experiment 2)
    print("\n[INFO] Running GroupBy PULocationID...")
    # We use a trigger action like show(5) to execute the plan
    top_pickups = df_clean.groupBy("PULocationID").count().orderBy(F.desc("count")).limit(5)
    print("Top 5 Pickup Locations:")
    top_pickups.show()

    t1 = time.time()
    print(f"\n[INFO] Demo completed in {t1 - t0:.2f} seconds.")

    # Stop Spark
    spark.stop()
    print("[INFO] Spark Session stopped.")


if __name__ == "__main__":
    import sys

    main()