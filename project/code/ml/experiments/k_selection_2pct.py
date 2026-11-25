# Location: project/code/ml/experiments/k_selection_2pct.py | PROJECT_ROOT depth: 3
# (formerly project/code/ml/k_selection_2pct.py)
# (archived method-selection experiment; not part of the production chain)
import os
import time
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# ============ Configuration ============
# Location: ml/experiments/k_selection_2pct.py -> 3 levels up = project/
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "output", "sampled_km")
SEED = 42
SUB_SAMPLE_FRACTION = 0.02
K_VALUES = [3, 4, 5]
MAX_ITER = 20

FEATURE_COLS = [
    "log_miles", "log_fare", "log_tips", "speed_mph", "log_duration",
    "hour_sin", "hour_cos", "day_sin", "day_cos"
]

# ============ Spark Session ============
spark = (SparkSession.builder
         .appName("KSelection_2percent")
         .master("local[*]")
         .config("spark.driver.memory", "8g")
         .config("spark.memory.offHeap.enabled", "true")
         .config("spark.memory.offHeap.size", "2g")
         .config("spark.sql.shuffle.partitions", "60")
         .config("spark.default.parallelism", "60")
         .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
         .config("spark.sql.adaptive.enabled", "true")
         .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
         .getOrCreate())

# ============ Load Full Data and Sample 2% ============
print("Loading full 10% sampled data...")
df_full = spark.read.parquet(INPUT_PATH)
total_count = df_full.count()
print(f"Full data count: {total_count:,}")

print(f"Sampling {SUB_SAMPLE_FRACTION*100:.0f}% (seed={SEED})...")
df_sample = df_full.sample(withReplacement=False, fraction=SUB_SAMPLE_FRACTION, seed=SEED)
sample_count = df_sample.count()
print(f"Sample count: {sample_count:,}")

# ============ Feature Engineering & Standardization ============
print("Assembling features and standardizing...")
assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="raw_features")
df_assembled = assembler.transform(df_sample)

scaler = StandardScaler(inputCol="raw_features", outputCol="features",
                        withMean=True, withStd=True)
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)
df_scaled.cache()
df_scaled.count()  # materialize cache
print("Standardization done. Cached.")

# ============ Evaluate K Values (Serial) ============
print("\n===== K Selection Results =====")
print(f"{'K':<6} {'WSSSE':<16} {'Silhouette':<12} {'Time (s)':<10}")
print("-" * 48)

results = []
for k in K_VALUES:
    start_time = time.time()

    bkm = BisectingKMeans(featuresCol="features", k=k, maxIter=MAX_ITER, seed=SEED)
    model = bkm.fit(df_scaled)

    wssse = model.computeCost(df_scaled)

    evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette",
                                    distanceMeasure="squaredEuclidean")
    silhouette = evaluator.evaluate(model.transform(df_scaled))

    elapsed = time.time() - start_time

    print(f"{k:<6} {wssse:<16.2f} {silhouette:<12.4f} {elapsed:<10.1f}")
    results.append((k, wssse, silhouette, elapsed))

# ============ Summary ============
print("\n===== Summary =====")
for k, wssse, sil, t in results:
    print(f"K={k}: WSSSE={wssse:.2f}, Silhouette={sil:.4f}, Time={t:.1f}s")

# ============ Cleanup ============
df_scaled.unpersist()
print("\nDone. Spark session will stop.")
spark.stop()