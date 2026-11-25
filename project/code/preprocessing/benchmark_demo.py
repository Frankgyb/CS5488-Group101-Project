"""
CS5488 Demo Benchmark – Quick validation on one month of data.
Tests partition size and core scaling with minimal runtime (~2-5 min).
"""

import time
import os
from datetime import datetime
from pyspark.sql import SparkSession

# ---------- Paths ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(PROJECT_ROOT, 'data')):
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

# Use only January 2023 data for fast demo
DEMO_DATA = os.path.join(PROJECT_ROOT, "data", "output", "cleaned",
                         "pickup_year=2023", "pickup_month=1")
LOG_DIR = os.path.join(PROJECT_ROOT, "data", "log")
os.makedirs(LOG_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"benchmark_demo_{TIMESTAMP}.txt")

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

def run_once(master, partition_bytes, columns=None):
    """Run a single benchmark trial and return elapsed seconds."""
    spark = SparkSession.builder \
        .appName("DemoBenchmark") \
        .master(master) \
        .config("spark.driver.memory", "12g") \
        .config("spark.sql.files.maxPartitionBytes", partition_bytes) \
        .config("spark.sql.shuffle.partitions", "10") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.local.dir", "D:/temp_spark") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    t0 = time.time()
    if columns:
        df = spark.read.parquet(DEMO_DATA).select(*columns)
    else:
        df = spark.read.parquet(DEMO_DATA)
    result = df.agg({"wait_minutes": "avg"}).collect()[0][0]
    elapsed = time.time() - t0
    spark.stop()
    return elapsed

if __name__ == "__main__":
    log("=" * 55)
    log(f"Demo Benchmark started at {datetime.now()}")
    log(f"Data: one month (2023-01), path: {DEMO_DATA}")
    log("Hardware: i7-12700H, 32GB RAM, SSD (assumed)")
    log("=" * 55)

    # ----- Experiment A: Partition size (fixed cores=4) -----
    log("\n--- A: Partition Size (local[4]) ---")
    for ps in ["32m", "128m", "512m"]:
        t = run_once("local[4]", ps)
        log(f"  Partition {ps}: {t:.2f}s")

    # ----- Experiment B: Core scaling (fixed partition=128MB) -----
    log("\n--- B: Core Scaling (128MB partition) ---")
    for cores in [1, 2, 4]:
        t = run_once(f"local[{cores}]", "128m")
        log(f"  Cores={cores}: {t:.2f}s")

    # ----- Experiment C: Column pruning (optional, fixed cores=4, partition=128MB) -----
    log("\n--- C: Column Pruning (local[4], 128MB) ---")
    t_all = run_once("local[4]", "128m")
    t_sub = run_once("local[4]", "128m", columns=["wait_minutes"])
    log(f"  All columns: {t_all:.2f}s")
    log(f"  Only wait_minutes: {t_sub:.2f}s")
    log(f"  Speedup: {t_all/t_sub:.2f}x" if t_sub > 0 else "  Speedup: N/A")

    log("\n" + "=" * 55)
    log(f"Demo finished at {datetime.now()}, log saved to {LOG_FILE}")