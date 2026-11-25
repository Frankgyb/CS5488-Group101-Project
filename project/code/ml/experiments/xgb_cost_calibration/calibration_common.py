# Location: project/code/ml/experiments/xgb_cost_calibration/calibration_common.py | PROJECT_ROOT depth: 4
"""Shared helpers for the XGBoost cost-calibration probes.

Scope: these scripts **measure how expensive a supervised stage would be**; they do not
train a deliverable model and they produce no model artifact. They exist so every number
quoted in the report's future-work section can be regenerated on this machine.

All paths are resolved from this file, never from the current working directory, so the
probes can be launched from the repository root, from ``project/`` or from this folder.
Outputs are confined to ``project/data/output/xgb_cost_calibration`` and
``project/data/log/xgb_cost_calibration``.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]  # .../project/
REPO_ROOT = PROJECT_ROOT.parent  # .../CS5488project_Group101/

CLEAN_PATH = PROJECT_ROOT / "data" / "output" / "cleaned"
OUT_ROOT = PROJECT_ROOT / "data" / "output" / "xgb_cost_calibration"
METRICS_DIR = OUT_ROOT / "metrics"
SAMPLE_DIR = OUT_ROOT / "sample5pct"
LOG_DIR = PROJECT_ROOT / "data" / "log" / "xgb_cost_calibration"
SPARK_LOCAL_DIR = OUT_ROOT / "_spark_tmp"

SEED = 42
SAMPLE_FRACTION = 0.05

# Whitelist used in the report's future-work section (engineered
# features only; raw tips/driver_pay and cluster/anomaly columns are excluded to avoid
# label leakage).
FEATURES = [
    "log_miles", "log_fare", "log_duration", "speed_mph",
    "fare_per_mile", "fare_per_min",
    "trip_miles", "trip_time", "base_passenger_fare",
    "tolls", "bcf", "sales_tax", "congestion_surcharge", "airport_fee",
    "wait_clipped", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "pickup_month", "PULocationID", "DOLocationID",
    "is_shared_request", "is_shared_match", "is_wav_request", "is_wav_match",
]
LABEL = "label_tip_ratio"
LABEL_BINARY = "label_zero_tip"

# v3 session configuration, copied from
# project/code/ml/experiments/parallel_scaling/scaling_common.py:265-276 so the
# calibration runs under the same Spark settings that the scaling study validated.
SPARK_CONF = {
    "spark.driver.memory": "8g",
    "spark.driver.memoryOverhead": "2g",
    "spark.memory.fraction": "0.6",
    "spark.memory.storageFraction": "0.3",
    "spark.memory.offHeap.enabled": "false",
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
    "spark.executor.extraJavaOptions": "-XX:+UseG1GC",
    "spark.ui.showConsoleProgress": "false",
}


# ---------------------------------------------------------------- filesystem helpers

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def month_dirs() -> list[Path]:
    """All ``pickup_year=*/pickup_month=*`` partitions of the cleaned table."""
    dirs = sorted(CLEAN_PATH.glob("pickup_year=*/pickup_month=*"))
    if not dirs:
        raise FileNotFoundError(f"no cleaned partitions under {CLEAN_PATH}")
    return dirs


# ------------------------------------------------------------------- logging / output

class Tee:
    """Mirror stdout lines to a log file without pulling in logging config."""

    def __init__(self, log_path: Path) -> None:
        ensure_dir(log_path.parent)
        self.handle = log_path.open("w", encoding="utf-8")

    def __call__(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def open_tee(name: str) -> Tee:
    return Tee(LOG_DIR / f"{name}.log")


def env_meta(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "timestamp": iso_now(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "cpu_count": os.cpu_count(),
        "argv": sys.argv,
        "project_root": str(PROJECT_ROOT),
    }
    for mod in ("pyspark", "xgboost", "numpy", "pyarrow"):
        try:
            meta[mod] = __import__(mod).__version__
        except Exception as exc:  # pragma: no cover - optional dependency
            meta[mod] = f"unavailable: {type(exc).__name__}"
    if extra:
        meta.update(extra)
    return meta


def write_metrics(name: str, payload: dict[str, Any]) -> Path:
    ensure_dir(METRICS_DIR)
    path = METRICS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def timed(label: str, fn: Callable[[], Any], log: Tee | None = None) -> tuple[float, Any]:
    start = time.perf_counter()
    result = fn()
    seconds = time.perf_counter() - start
    message = f"[{label}] {seconds:.1f} s -> {result}"
    log(message) if log else print(message, flush=True)
    return seconds, result


# ------------------------------------------------------------------------- Spark bits

def build_spark(master: str = "local[*]", shuffle: int = 120, tag: str = "xgbcal"):
    """Spark session using the v3 configuration (see SPARK_CONF)."""
    ensure_dir(SPARK_LOCAL_DIR)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName(f"XgbCostCalibration-{tag}").master(master)
    for key, value in SPARK_CONF.items():
        builder = builder.config(key, value)
    spark = (
        builder
        .config("spark.local.dir", str(SPARK_LOCAL_DIR))
        .config("spark.sql.shuffle.partitions", str(shuffle))
        .config("spark.default.parallelism", str(shuffle))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def feature_frame(df):
    """Feature derivation mirroring plan §3.1; also emits both labels."""
    from pyspark.sql import functions as F

    eps = 1e-9
    base = df.select(
        "pickup_datetime", "PULocationID", "DOLocationID",
        "trip_miles", "trip_time", "trip_duration_minutes",
        "base_passenger_fare", "tolls", "bcf", "sales_tax",
        "congestion_surcharge", "airport_fee", "tips",
        "wait_minutes", "pickup_hour", "pickup_day_of_week",
        "is_shared_request", "is_shared_match", "is_wav_request", "is_wav_match",
    )
    return (
        base
        .withColumn(
            LABEL,
            F.when(F.col("base_passenger_fare") > 0,
                   F.least(F.col("tips") / F.col("base_passenger_fare"), F.lit(1.0))),
        )
        .withColumn(LABEL_BINARY, (F.col("tips") == 0).cast("int"))
        .withColumn("log_miles", F.log1p(F.col("trip_miles")))
        .withColumn("log_fare", F.log1p(F.col("base_passenger_fare")))
        .withColumn("log_duration", F.log1p(F.col("trip_duration_minutes")))
        .withColumn("speed_mph",
                    F.col("trip_miles") / (F.col("trip_duration_minutes") / 60.0 + eps))
        .withColumn("fare_per_mile",
                    F.col("base_passenger_fare") / (F.col("trip_miles") + 0.01))
        .withColumn("fare_per_min",
                    F.col("base_passenger_fare") / (F.col("trip_duration_minutes") + 0.01))
        .withColumn("hour_sin",
                    F.sin(2 * 3.141592653589793 * F.col("pickup_hour") / 24))
        .withColumn("hour_cos",
                    F.cos(2 * 3.141592653589793 * F.col("pickup_hour") / 24))
        .withColumn("dow_sin",
                    F.sin(2 * 3.141592653589793 * F.col("pickup_day_of_week") / 7))
        .withColumn("dow_cos",
                    F.cos(2 * 3.141592653589793 * F.col("pickup_day_of_week") / 7))
        .withColumn("pickup_month", F.month("pickup_datetime"))
        .withColumn("wait_clipped",
                    F.least(F.greatest(F.col("wait_minutes"), F.lit(0.0)), F.lit(120.0)))
    )


def valid_filter():
    """Row filter from plan §3.2 (positive fare/miles/time, plausible speed, label set)."""
    from pyspark.sql import functions as F

    return (
        (F.col("trip_miles") > 0)
        & (F.col("trip_duration_minutes") > 0)
        & (F.col("base_passenger_fare") > 0)
        & (F.col("speed_mph") <= 100)
        & (F.col(LABEL).isNotNull())
    )


def add_common_args(parser, *, master_default: str = "local[*]", shuffle_default: int = 120):
    parser.add_argument("--master", default=master_default,
                        help="Spark master, e.g. local[4] (default: local[*])")
    parser.add_argument("--shuffle-partitions", type=int, default=shuffle_default,
                        help="spark.sql.shuffle.partitions (default: 120, as in v3)")
    return parser


def feature_array(tbl, columns: Iterable[str]):
    """Arrow table -> float32 matrix without pandas round-trips."""
    import numpy as np

    n = tbl.num_rows
    matrix = np.empty((n, len(list(columns))), dtype=np.float32)
    for i, column in enumerate(columns):
        matrix[:, i] = tbl.column(column).to_numpy(zero_copy_only=False).astype(
            np.float32, copy=False)
    return matrix
