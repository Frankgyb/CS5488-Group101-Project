"""
Full Benchmark – All available NYC FHVHV data (approx 11.6 GB, 472 million rows).
Tests partition size, core scaling, and column pruning with full dataset.
Expected total runtime: 15–30 minutes (depending on disk speed and CPU).
Results are saved to data/log/benchmark_full_{timestamp}.txt.
"""

import time
import os
from datetime import datetime
from functools import reduce
from pyspark.sql import SparkSession

# ---------- Paths ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(PROJECT_ROOT, 'data')):
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "output", "cleaned")
LOG_DIR = os.path.join(PROJECT_ROOT, "data", "log")
os.makedirs(LOG_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"benchmark_full_{TIMESTAMP}.txt")

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

def discover_partitions(base_dir):
    """Discover all year/month partitions under cleaned directory."""
    partitions = []
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Cleaned directory not found: {base_dir}")
    for year_dir in sorted(os.listdir(base_dir)):
        year_path = os.path.join(base_dir, year_dir)
        if not os.path.isdir(year_path) or not year_dir.startswith("pickup_year="):
            continue
        for month_dir in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month_dir)
            if os.path.isdir(month_path) and month_dir.startswith("pickup_month="):
                partitions.append(month_path)
    if not partitions:
        raise RuntimeError("No partitions found under cleaned directory.")
    log(f"Discovered {len(partitions)} partitions (year/month).")
    return partitions

def run_experiment(master, partition_bytes, columns=None, repeat=3):
    """
    Run a single experiment configuration.
    Reads all discovered partitions, applies optional column selection,
    computes average wait_minutes, returns average elapsed time over `repeat` runs.
    """
    partitions = discover_partitions(CLEANED_DIR)
    times = []
    for i in range(repeat):
        spark = SparkSession.builder \
            .appName(f"FullBench-{master}-{partition_bytes}") \
            .master(master) \
            .config("spark.driver.memory", "12g") \
            .config("spark.sql.files.maxPartitionBytes", partition_bytes) \
            .config("spark.sql.shuffle.partitions", "40") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.local.dir", "D:/temp_spark") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")

        t0 = time.time()
        # Read all partitions and union them
        dfs = [spark.read.parquet(p) for p in partitions]
        df = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), dfs)
        if columns:
            df = df.select(*columns)
        # Force action: compute average wait_minutes
        result = df.agg({"wait_minutes": "avg"}).collect()[0][0]
        elapsed = time.time() - t0
        spark.stop()
        times.append(elapsed)
        log(f"  Run {i+1}: {elapsed:.2f}s")
    avg = sum(times) / len(times)
    log(f"  Average over {repeat} runs: {avg:.2f}s\n")
    return avg

if __name__ == "__main__":
    log("=" * 65)
    log(f"Full Benchmark started at {datetime.now()}")
    log(f"Cleaned data root: {CLEANED_DIR}")
    log("Hardware: i7-12700H, 32GB RAM, SSD (assumed)")
    log("=" * 65)

    # ----- Experiment A: Partition size (fixed cores=4) -----
    log("\n--- A: Partition Size (local[4]) ---")
    for ps in ["32m", "128m", "512m", "2048m"]:
        log(f"Partition {ps}:")
        run_experiment("local[4]", ps)

    # ----- Experiment B: Core scaling (fixed partition=128MB) -----
    log("\n--- B: Core Scaling (128MB partition) ---")
    for cores in [1, 2, 4, 8]:
        log(f"Cores={cores}:")
        run_experiment(f"local[{cores}]", "128m")

    # ----- Experiment C: Column pruning (fixed cores=4, partition=128MB) -----
    log("\n--- C: Column Pruning (local[4], 128MB) ---")
    log("All columns:")
    t_all = run_experiment("local[4]", "128m")
    log("Only wait_minutes, base_passenger_fare, trip_miles:")
    t_sub = run_experiment("local[4]", "128m",
                           columns=["wait_minutes", "base_passenger_fare", "trip_miles"])
    speedup = t_all / t_sub if t_sub > 0 else float('inf')
    log(f"Column pruning speedup: {speedup:.2f}x")

    log("\n" + "=" * 65)
    log(f"Full Benchmark finished at {datetime.now()}")
    log(f"Results saved to {LOG_FILE}")