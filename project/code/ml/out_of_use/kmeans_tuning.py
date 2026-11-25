"""
Location: project/code/ml/out_of_use/kmeans_tuning.py | PROJECT_ROOT depth: 3
(now bug-fixed to point at .../project; NOT part of the current pipeline - kept for reference)
KMeans tuning script with stratified sampling using real field names.
Reads cleaned Parquet, extracts month from pickup_datetime for stratification,
computes speed_mph from trip_miles / (trip_time / 3600),
runs KMeans for k=3..8, evaluates WSSSE + Silhouette, and saves results.

Hardware: 14-core CPU, 32GB RAM
Memory: 10g driver + 10g executor + 3g offHeap
Shuffle partitions: 14 (one per core)
Output: data/output/tuning_results/tuning_results.csv
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, when, month
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.storagelevel import StorageLevel
import pandas as pd
import os
import time

# ==================== Configuration ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Location: code/ml/out_of_use/kmeans_tuning.py -> 3 levels up = project/
# (之前是 2 级, 实际算到 code/ 而非 project/ —— 已修正)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

INPUT_PATH = os.path.join(PROJECT_ROOT, "data/output/cleaned/")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/output/tuning_results/")
BEST_K_FILE = os.path.join(PROJECT_ROOT, "data/output/best_k.txt")
SAMPLE_RATE = 0.1
SEED = 42
K_RANGE = range(3, 9)          # k = 3 to 8
MAX_ITER = 20
TOL = 1e-4

# Real feature columns confirmed from EDA log
FEATURE_COLS = ["trip_miles", "trip_time", "base_passenger_fare", "tips", "speed_mph"]
STRATIFICATION_COLS = ["PULocationID", "pickup_month"]  # derived from pickup_datetime

# ==================== Spark Session (Tuned for 14 cores, 32GB) ====================
spark = SparkSession.builder \
    .appName("KMeansTuning") \
    .master("local[*]") \
    .config("spark.driver.memory", "10g") \
    .config("spark.executor.memory", "10g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "3g") \
    .config("spark.sql.shuffle.partitions", "20") \
    .config("spark.default.parallelism", "20") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.shuffle.file.buffer", "64k") \
    .config("spark.local.dir", "D:\\temp_spark") \
    .getOrCreate()

start_time = time.time()

# ==================== Read & Filter ====================
print("Reading cleaned Parquet from:", INPUT_PATH)
df = spark.read.parquet(INPUT_PATH)

print("Filtering valid trips using real fields...")
# Use only real fields; filtering based on business rules
valid_df = df.filter(
    (col("trip_time") > 0) &
    (col("trip_miles") >= 0) &
    (col("base_passenger_fare") >= 0)
).select(
    col("trip_miles"),
    col("trip_time"),
    col("base_passenger_fare"),
    col("tips"),
    col("PULocationID"),
    col("pickup_datetime")       # use real timestamp column
)

# Extract month from pickup_datetime for stratification
valid_df = valid_df.withColumn("pickup_month", month(col("pickup_datetime")))

# Compute speed_mph = miles / (seconds / 3600)
valid_df = valid_df.withColumn(
    "speed_mph",
    when(col("trip_time") > 0,
         col("trip_miles") / (col("trip_time") / 3600.0)).otherwise(0.0)
)

# ==================== Stratified Sampling ====================
print("Building stratification key (PULocationID + pickup_month)...")
valid_df = valid_df.withColumn(
    "strata_key",
    concat_ws("_", col("PULocationID"), col("pickup_month"))
)

# Collect distinct strata keys efficiently
strata_keys = [r.strata_key for r in
               valid_df.select("strata_key").distinct().collect()]
fractions = {k: SAMPLE_RATE for k in strata_keys}
print(f"Number of strata: {len(strata_keys)}")

print(f"Sampling at rate={SAMPLE_RATE}...")
sampled_df = valid_df.sampleBy("strata_key", fractions, seed=SEED) \
                     .drop("strata_key", "pickup_datetime", "pickup_month", "PULocationID") \
                     .select(*FEATURE_COLS)

sampled_df.persist(StorageLevel.MEMORY_AND_DISK_DESER).count()
print(f"Sampled rows: {sampled_df.count():,}")

# ==================== Feature Engineering ====================
assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_unscaled")
feature_vec = assembler.transform(sampled_df)

scaler = StandardScaler(inputCol="features_unscaled", outputCol="features",
                        withStd=True, withMean=True)
scaler_model = scaler.fit(feature_vec)
scaled_df = scaler_model.transform(feature_vec)

# ==================== Tuning Loop ====================
results = []
best_k = None
best_sil = -1.0

for k in K_RANGE:
    print(f"\n--- k={k} ---")
    kmeans = KMeans(featuresCol="features", predictionCol="prediction",
                    k=k, initMode="k-means||", maxIter=MAX_ITER,
                    tol=TOL, seed=SEED)
    model = kmeans.fit(scaled_df)

    wssse = model.summary.trainingCost
    predictions = model.transform(scaled_df)

    evaluator = ClusteringEvaluator(featuresCol="features",
                                    metricName="silhouette",
                                    distanceMeasure="squaredEuclidean")
    sil = evaluator.evaluate(predictions)

    elapsed_k = time.time() - start_time
    print(f"  WSSSE={wssse:.2f}  Silhouette={sil:.4f}  [{elapsed_k:.1f}s elapsed]")
    results.append((k, wssse, sil))

    if sil > best_sil:
        best_sil = sil
        best_k = k

total_elapsed = time.time() - start_time
print(f"\nBest k = {best_k}  (Silhouette = {best_sil:.4f})")
print(f"Total time: {total_elapsed/60:.2f} min")

# ==================== Save Results ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)
pdf = pd.DataFrame(results, columns=["k", "wssse", "silhouette"])
pdf["elapsed_sec"] = round(total_elapsed, 1)
out_path = os.path.join(OUTPUT_DIR, "tuning_results.csv")
pdf.to_csv(out_path, index=False)
print(f"Results saved to {out_path}")

with open(BEST_K_FILE, "w") as f:
    f.write(str(best_k))
print(f"Best k saved to {BEST_K_FILE}")

# ==================== Cleanup ====================
sampled_df.unpersist()
spark.stop()
print("Done.")