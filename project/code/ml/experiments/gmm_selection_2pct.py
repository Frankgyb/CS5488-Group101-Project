# Location: project/code/ml/experiments/gmm_selection_2pct.py | PROJECT_ROOT depth: 3
# (formerly project/code/ml/gmm_selection_2pct.py)
# (archived method-selection experiment; not part of the production chain)
import os
import sys
import time
import logging
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import GaussianMixture

# ============ Configuration ============
# Location: ml/experiments/gmm_selection_2pct.py -> 3 levels up = project/
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
LOG_DIR = os.path.join(PROJECT_ROOT, "data", "log")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"gmm_selection_{time.strftime('%Y%m%d_%H%M%S')}.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
logger.info("gmm_selection_2pct start | PROJECT_ROOT=%s", PROJECT_ROOT)
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "output", "sampled_km")
SEED = 42
SUB_SAMPLE_FRACTION = 0.02
K_VALUES = [2, 3, 4, 5]
MAX_ITER = 20
TOL = 1e-4

FEATURE_COLS = [
    "log_miles", "log_fare", "log_tips", "speed_mph", "log_duration",
    "hour_sin", "hour_cos", "day_sin", "day_cos"
]

# ============ Spark Session ============
spark = (SparkSession.builder
         .appName("GMM_Selection_2percent")
         .master("local[*]")
         .config("spark.driver.memory", "10g")
         .config("spark.memory.offHeap.enabled", "true")
         .config("spark.memory.offHeap.size", "2g")
         .config("spark.sql.shuffle.partitions", "60")
         .config("spark.default.parallelism", "60")
         .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
         .config("spark.sql.adaptive.enabled", "true")
         .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
         .getOrCreate())

# ============ Load & Sample ============
print("Loading full 10% sampled data...")
df_full = spark.read.parquet(INPUT_PATH)
total_count = df_full.count()
print(f"Full data count: {total_count:,}")

print(f"Sampling {SUB_SAMPLE_FRACTION*100:.0f}% (seed={SEED})...")
df_sample = df_full.sample(withReplacement=False, fraction=SUB_SAMPLE_FRACTION, seed=SEED)
sample_count = df_sample.count()
print(f"Sample count: {sample_count:,}")

# ============ Feature Engineering ============
print("Assembling features and standardizing...")
assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="raw_features")
df_assembled = assembler.transform(df_sample)

scaler = StandardScaler(inputCol="raw_features", outputCol="features",
                        withMean=True, withStd=True)
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)
df_scaled.cache()
df_scaled.count()
print("Standardization done. Cached.")

# ============ Evaluate K Values with GMM ============
print("\n===== GMM K Selection Results =====")
print(f"{'K':<6} {'LogLikelihood':<20} {'BIC':<18} {'Time (s)':<10}")
print("-" * 56)

results = []
d = len(FEATURE_COLS)  # number of features

for k in K_VALUES:
    start_time = time.time()

    gmm = GaussianMixture(k=k, featuresCol="features", maxIter=MAX_ITER,
                          tol=TOL, seed=SEED)
    model = gmm.fit(df_scaled)

    log_likelihood = model.summary.logLikelihood

    # Manual BIC calculation
    # Number of parameters:
    #   - covariance matrices: k * d * (d+1) / 2
    #   - means: k * d
    #   - weights: k - 1
    num_params = k * d * (d + 1) // 2 + k * d + (k - 1)
    bic = -2 * log_likelihood + num_params * np.log(sample_count)

    elapsed = time.time() - start_time

    print(f"{k:<6} {log_likelihood:<20.2f} {bic:<18.2f} {elapsed:<10.1f}")
    results.append((k, log_likelihood, bic, elapsed))

# ============ Summary ============
print("\n===== Summary =====")
print(f"{'K':<6} {'BIC':<18} {'LogLikelihood':<20} {'Time':<10}")
print("-" * 56)
for k, ll, bic, t in results:
    print(f"{k:<6} {bic:<18.2f} {ll:<20.2f} {t:<10.1f}")

best_k = min(results, key=lambda x: x[2])[0]
print(f"\nRecommended K (lowest BIC): {best_k}")

# ============ Cleanup ============
df_scaled.unpersist()
print("\nDone. Spark session will stop.")
spark.stop()