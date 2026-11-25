# Location: project/code/ml/experiments/xgb_cost_calibration/10_scan_and_sample.py | PROJECT_ROOT depth: 4
"""Probe A: cost of the Spark feature pass over the cleaned table, and of materialising
a stratified L1 sample to Parquet.

Measures (all wall-clock, single run):
  1. warm-up count on one month,
  2. full feature build + filter over two months,
  3. full feature build + filter over all 24 months,
  4. feature build + filter + 5 % sample write to Parquet.

Writes ``data/output/xgb_cost_calibration/metrics/scan_and_sample.json`` and the sample
under ``data/output/xgb_cost_calibration/sample5pct`` (≈1.3 GB, regenerable, excluded
from the submission package — see the output README).

Usage:
    python project/code/ml/experiments/xgb_cost_calibration/10_scan_and_sample.py
    ... --skip-write          # scan only, keep the existing sample
    ... --sample-fraction 0.02
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration_common import (  # noqa: E402
    CLEAN_PATH, SAMPLE_DIR, SAMPLE_FRACTION, SEED, add_common_args, build_spark,
    ensure_dir, env_meta, feature_frame, month_dirs, open_tee, valid_filter,
    write_metrics,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--sample-fraction", type=float, default=SAMPLE_FRACTION,
                        help=f"sample fraction for the L1 sample (default {SAMPLE_FRACTION})")
    parser.add_argument("--skip-write", action="store_true",
                        help="do not rewrite the sample (timing of the write is then absent)")
    args = parser.parse_args(argv)

    log = open_tee("10_scan_and_sample")
    spark = build_spark(args.master, args.shuffle_partitions, tag="scan")
    log(f"master={spark.sparkContext.master} cores={spark.sparkContext.defaultParallelism}")
    result: dict = {"stage": "scan_and_sample"}

    try:
        from pyspark.sql import functions as F

        months = month_dirs()
        log(f"cleaned partitions: {len(months)} under {CLEAN_PATH}")
        result["n_month_partitions"] = len(months)

        # 1) warm-up on a single month
        warm_month = CLEAN_PATH / "pickup_year=2023" / "pickup_month=1"
        start = time.perf_counter()
        n_warm = spark.read.parquet(str(warm_month)).count()
        result["warmup_s"] = round(time.perf_counter() - start, 1)
        result["warmup_rows"] = n_warm
        log(f"warmup {result['warmup_s']} s, {n_warm:,} rows")

        def counted(frame):
            return feature_frame(frame).filter(valid_filter()).count()

        # 2) two months
        two = [CLEAN_PATH / "pickup_year=2023" / "pickup_month=1",
               CLEAN_PATH / "pickup_year=2023" / "pickup_month=2"]
        start = time.perf_counter()
        n_two = counted(spark.read.parquet(*[str(p) for p in two]))
        result["two_months_s"] = round(time.perf_counter() - start, 1)
        result["two_months_rows"] = n_two
        result["two_months_rows_per_s"] = round(n_two / max(result["two_months_s"], 1e-9))
        log(f"two months {result['two_months_s']} s, {n_two:,} rows")

        # 3) all months
        start = time.perf_counter()
        n_all = counted(spark.read.parquet(str(CLEAN_PATH)))
        result["all_months_s"] = round(time.perf_counter() - start, 1)
        result["all_months_rows"] = n_all
        result["all_months_rows_per_s"] = round(n_all / max(result["all_months_s"], 1e-9))
        log(f"all months {result['all_months_s']} s, {n_all:,} rows")

        # 4) sample write
        if args.skip_write:
            log("skipping sample write (--skip-write)")
        else:
            start = time.perf_counter()
            frame = feature_frame(spark.read.parquet(str(CLEAN_PATH))).filter(valid_filter())
            sample = frame.sample(withReplacement=False,
                                  fraction=args.sample_fraction, seed=SEED)
            ensure_dir(SAMPLE_DIR.parent)
            sample.write.mode("overwrite").parquet(str(SAMPLE_DIR))
            n_sample = spark.read.parquet(str(SAMPLE_DIR)).count()
            result["sample_write_s"] = round(time.perf_counter() - start, 1)
            result["sample_rows"] = n_sample
            result["sample_fraction"] = args.sample_fraction
            result["sample_dir"] = str(SAMPLE_DIR)
            result["sample_files"] = len(list(SAMPLE_DIR.glob("part-*.parquet")))
            bytes_total = sum(p.stat().st_size for p in SAMPLE_DIR.rglob("*.parquet"))
            result["sample_bytes"] = bytes_total
            result["sample_gb"] = round(bytes_total / 1e9, 2)
            log(f"sample write {result['sample_write_s']} s, {n_sample:,} rows, "
                f"{result['sample_gb']} GB, {result['sample_files']} files")

        # sanity: label distribution of the sample, useful when quoting it
        if not args.skip_write:
            sample_df = spark.read.parquet(str(SAMPLE_DIR))
            dist = {
                row["label"]: row["n"]
                for row in sample_df.groupBy(F.col("label_zero_tip").alias("label"))
                .agg(F.count(F.lit(1)).alias("n")).collect()
            }
            total = sum(dist.values()) or 1
            result["sample_zero_tip_share"] = round(dist.get(1, 0) / total, 4)
            log(f"sample zero-tip share {result['sample_zero_tip_share']}")

        result["env"] = env_meta()
        path = write_metrics("scan_and_sample", result)
        log(f"metrics -> {path}")
        return 0
    finally:
        spark.stop()
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
