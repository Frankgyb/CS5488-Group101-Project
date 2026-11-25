# Location: project/code/ml/experiments/xgb_cost_calibration/12_spark_ml_probe.py | PROJECT_ROOT depth: 4
"""Probe C: price of the JVM-side Spark tree models (route P3 in the future-work note).

``pyspark.ml`` GBT/RF execute entirely inside the JVM, so they keep working even though
this PySpark build cannot start a Python worker (see 13_python_worker_probe.py). This
script measures their per-iteration / per-tree cost on a row slice of the L1 sample, which
is what a distributed baseline would cost.

Writes ``metrics/spark_ml_probe.json``.

Usage:
    python project/code/ml/experiments/xgb_cost_calibration/12_spark_ml_probe.py
    ... --rows 2000000 --master local[8] --gbt-iters 5 --rf-trees 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration_common import (  # noqa: E402
    FEATURES, LABEL, SAMPLE_DIR, add_common_args, build_spark, env_meta, open_tee,
    write_metrics,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, master_default="local[8]")
    parser.add_argument("--rows", type=int, default=2_000_000,
                        help="rows to use (default 2 M)")
    parser.add_argument("--gbt-iters", type=int, default=5)
    parser.add_argument("--gbt-depth", type=int, default=4)
    parser.add_argument("--rf-trees", type=int, default=10)
    parser.add_argument("--rf-depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    log = open_tee("12_spark_ml_probe")
    if not SAMPLE_DIR.is_dir():
        log(f"ERROR: {SAMPLE_DIR} missing — run 10_scan_and_sample.py first")
        return 2

    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
    from pyspark.sql import functions as F

    spark = build_spark(args.master, args.shuffle_partitions, tag="ml")
    result: dict = {"stage": "spark_ml_probe",
                    "master": spark.sparkContext.master,
                    "cores": spark.sparkContext.defaultParallelism}
    log(f"master={result['master']} cores={result['cores']}")

    try:
        start = time.perf_counter()
        frame = (
            spark.read.parquet(str(SAMPLE_DIR))
            .limit(args.rows)
            .select(FEATURES + [LABEL])
        )
        frame = VectorAssembler(inputCols=FEATURES, outputCol="features").transform(frame)
        frame = frame.select("features", F.col(LABEL).alias("label")).cache()
        n_rows = frame.count()
        result["read_assemble_s"] = round(time.perf_counter() - start, 1)
        result["n_rows"] = n_rows
        log(f"read+assemble+cache {n_rows:,} rows in {result['read_assemble_s']} s")

        if args.gbt_iters:
            start = time.perf_counter()
            GBTRegressor(featuresCol="features", labelCol="label",
                         maxIter=args.gbt_iters, maxDepth=args.gbt_depth,
                         seed=args.seed).fit(frame)
            gbt_s = time.perf_counter() - start
            result["gbt"] = {
                "max_iter": args.gbt_iters,
                "max_depth": args.gbt_depth,
                "fit_s": round(gbt_s, 1),
                "s_per_iter": round(gbt_s / args.gbt_iters, 2),
            }
            log(f"GBT(maxIter={args.gbt_iters}, maxDepth={args.gbt_depth}): "
                f"{gbt_s:.1f} s ({gbt_s / args.gbt_iters:.2f} s/iter)")

        if args.rf_trees:
            start = time.perf_counter()
            RandomForestRegressor(featuresCol="features", labelCol="label",
                                  numTrees=args.rf_trees, maxDepth=args.rf_depth,
                                  seed=args.seed).fit(frame)
            rf_s = time.perf_counter() - start
            result["rf"] = {
                "num_trees": args.rf_trees,
                "max_depth": args.rf_depth,
                "fit_s": round(rf_s, 1),
                "s_per_tree": round(rf_s / args.rf_trees, 2),
            }
            log(f"RF(numTrees={args.rf_trees}, maxDepth={args.rf_depth}): "
                f"{rf_s:.1f} s ({rf_s / args.rf_trees:.2f} s/tree)")

        result["env"] = env_meta()
        log(f"metrics -> {write_metrics('spark_ml_probe', result)}")
        return 0
    finally:
        spark.stop()
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
