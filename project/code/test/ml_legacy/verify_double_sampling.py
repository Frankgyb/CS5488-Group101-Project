# Location: project/code/test/ml_legacy/verify_double_sampling.py | PROJECT_ROOT depth: 3
# (formerly project/code/ml/verify_double_sampling.py - sampling audit, not part of the main pipeline)
import os
import math
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean as spark_mean

# ============ Configuration ============
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Location: code/test/ml_legacy/verify_double_sampling.py -> 4 levels up = project/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))  # .../project

os.environ["HADOOP_HOME"] = r"D:\hadoop"
os.environ["hadoop.home.dir"] = r"D:\hadoop"

INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "output", "sampled_km")
SEED = 42
SUB_SAMPLE_RATIO = 0.02  # 2%  (modified from 0.5%)

FEATURE_COLS = [
    "log_miles", "log_fare", "log_tips", "speed_mph", "log_duration",
    "hour_sin", "hour_cos", "day_sin", "day_cos"
]

# ============ Spark Session ============
spark = (SparkSession.builder
         .appName("VerifyDoubleSampling")
         .master("local[*]")
         .config("spark.driver.memory", "14g")
         .config("spark.memory.offHeap.enabled", "true")
         .config("spark.memory.offHeap.size", "2g")
         .config("spark.sql.shuffle.partitions", "64")
         .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
         .config("spark.local.dir", "D:/temp_spark")
         .getOrCreate())

# ============ Load 10% Sampled Data ============
print("Loading 10% sampled data...")
sampled_10pct = spark.read.parquet(INPUT_PATH)
count_10pct = sampled_10pct.count()
print(f"10% sampled data count: {count_10pct:,}")

# ============ Extract 2% Sub-sample (Stratified by bucket) ============
print("Extracting 2% sub-sample (stratified by bucket)...")
# Build fraction dictionary: same ratio for every bucket
bucket_list = sampled_10pct.select("bucket").distinct().collect()
layer_fractions = {row["bucket"]: SUB_SAMPLE_RATIO for row in bucket_list}
sub_sample = sampled_10pct.sampleBy("bucket", fractions=layer_fractions, seed=SEED)
count_05pct = sub_sample.count()
print(f"2% sub-sample count: {count_05pct:,}")

# ============ Representativeness Check ============
print("\n===== Representativeness Report =====")
print(f"{'Feature':<12} {'10% Mean':<12} {'2% Mean':<12} {'Bias':<10} {'Status'}")
print("-" * 70)

for col_name in FEATURE_COLS:
    mean_10 = sampled_10pct.agg(spark_mean(col(col_name))).collect()[0][0]
    mean_05 = sub_sample.agg(spark_mean(col(col_name))).collect()[0][0]

    bias = abs(mean_10 - mean_05) / abs(mean_10) if mean_10 != 0 else 0
    status = "PASS" if bias < 0.01 else "WARN"

    print(f"{col_name:<12} {mean_10:<12.4f} {mean_05:<12.4f} {bias:<10.2%} {status}")

# ============ Layer Distribution Check ============
print("\n===== Layer Distribution Check =====")
print("10% sampled data layer distribution:")
sampled_10pct.groupBy("bucket").count().show()

print("2% sub-sample layer distribution:")
sub_sample.groupBy("bucket").count().show()

print("\nVerification complete.")
spark.stop()