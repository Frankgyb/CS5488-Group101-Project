# Location: project/code/ml/experiments/kmeans_full.py | PROJECT_ROOT depth: 3
# (formerly project/code/ml/kmeans_full.py - full-data KMeans experiment)
# (archived method-selection experiment; not part of the production chain)
# code/ml/kmeans_full.py
import os
import logging
import warnings
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Suppress harmless Python warnings (e.g., FutureWarning for computeCost)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============ Configuration ============
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Location: ml/experiments/kmeans_full.py -> 3 levels up = project/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))

INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "output", "sampled_km")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "output", "kmeans_full_result")
LOG_DIR = os.path.join(PROJECT_ROOT, "data", "log")
LOG_PATH = os.path.join(LOG_DIR, "kmeans_full.log")
BEST_K = 8
SEED = 42

FEATURE_COLS = [
    "log_miles", "log_fare", "log_tips", "speed_mph", "log_duration",
    "hour_sin", "hour_cos", "day_sin", "day_cos"
]

# ============ Logging Setup ============
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ============ Spark Session ============
spark = (SparkSession.builder
    .appName("KMeansFullData")
    .master("local[*]")
    .config("spark.local.dir", "D:/temp_spark")
    .config("spark.sql.shuffle.partitions", "30")
    .config("spark.default.parallelism", "30")
    .config("spark.driver.memory", "8g")
    .config("spark.driver.memoryOverhead", "2g")
    .config("spark.memory.fraction", "0.6")
    .config("spark.memory.storageFraction", "0.3")
    .config("spark.memory.offHeap.enabled", "false")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
    .getOrCreate())

# Set Spark log level to ERROR to suppress INFO/WARN messages
spark.sparkContext.setLogLevel("ERROR")

# ============ Load Data ============
logger.info("Loading full sampled_km data...")
df = spark.read.parquet(INPUT_PATH)
total_rows = df.count()
logger.info(f"Full data count: {total_rows:,}")

# ============ Assemble & Scale ============
logger.info("Assembling features and standardizing...")
assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_vec")
df_vec = assembler.transform(df)

scaler = StandardScaler(inputCol="features_vec", outputCol="scaled_features",
                         withMean=True, withStd=True)
scaler_model = scaler.fit(df_vec)
df_scaled = scaler_model.transform(df_vec)
df_scaled.cache()

# ============ Train BisectingKMeans ============
logger.info(f"Training BisectingKMeans with K={BEST_K}...")
bkm = BisectingKMeans(k=BEST_K, seed=SEED, featuresCol="scaled_features")
model = bkm.fit(df_scaled)

predictions = model.transform(df_scaled)

# ============ Evaluate ============
evaluator = ClusteringEvaluator(featuresCol="scaled_features")
silhouette = evaluator.evaluate(predictions)
cost = model.computeCost(df_scaled)

logger.info("\n===== Full Data Clustering Result =====")
logger.info(f"K={BEST_K}")
logger.info(f"Silhouette Score: {silhouette:.4f}")
logger.info(f"WSSSE (Cost): {cost:.2f}")

# ============ Save Results ============
persist_cols = FEATURE_COLS + [
    "PULocationID", "hvfhs_license_num", "bucket", "layer_key",
    "pickup_year", "pickup_month", "prediction"
]
output_df = predictions.select(*persist_cols)

os.makedirs(OUTPUT_PATH, exist_ok=True)
output_df.write.mode("overwrite").parquet(OUTPUT_PATH)
logger.info(f"Clustering results saved to {OUTPUT_PATH}")

logger.info("Done. Spark session will stop.")
spark.stop()