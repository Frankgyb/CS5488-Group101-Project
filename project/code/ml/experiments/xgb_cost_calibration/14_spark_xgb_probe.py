# Location: project/code/ml/experiments/xgb_cost_calibration/14_spark_xgb_probe.py | PROJECT_ROOT depth: 4
"""Probe E: unit cost of distributed XGBoost (route P1), now that it turns out to work.

Probe D showed the workers were never broken by the ``pyspark.daemon`` import; they were
killed by interpreter resolution (``python3`` missing), and ``PYSPARK_PYTHON`` fixes it. So
``xgboost.spark.SparkXGBRegressor`` — the route the plan called P1 — is available on this
machine. This probe prices it: one fit per worker count, each in a fresh session because
``spark.task.cpus`` cannot change at runtime.

Note that ``xgboost.spark`` ignores ``nthread``/``n_jobs`` (they are in
``_unsupported_xgb_params``), so per-worker threads always equal ``spark.task.cpus`` (1 here);
parallelism comes only from ``num_workers``.

Input is the 5 % sample written by ``10_scan_and_sample.py`` (23.6 M rows); ``--rows`` slices
it with a seeded fraction so every config trains on the same data. Writes
``metrics/spark_xgb_probe.json``.

Usage:
    python project/code/ml/experiments/xgb_cost_calibration/14_spark_xgb_probe.py
    ... --rows 2000000 --workers 4,8 --rounds 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from calibration_common import (  # noqa: E402
    FEATURES, LABEL, SAMPLE_DIR, SEED, SPARK_CONF, SPARK_LOCAL_DIR, add_common_args,
    ensure_dir, env_meta, open_tee, write_metrics,
)

CHILD = r"""
import json, sys, time, traceback
from pathlib import Path
cfg = json.loads(sys.argv[1])

def main():
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql import functions as F
    from xgboost.spark import SparkXGBRegressor

    builder = SparkSession.builder.appName("XgbCostCalibration-sparkxgb").master(cfg["master"])
    for key, value in cfg["conf"].items():
        builder = builder.config(key, value)
    spark = builder.config("spark.task.cpus", str(cfg["task_cpus"])).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    out = {"num_workers": cfg["num_workers"], "task_cpus": cfg["task_cpus"],
           "master": cfg["master"], "rows": cfg["rows"], "rounds": cfg["rounds"]}
    try:
        t0 = time.perf_counter()
        df = spark.read.parquet(cfg["sample_dir"]).select(cfg["features"] + [cfg["label"]])
        df = df.sample(fraction=cfg["fraction"], seed=cfg["seed"])
        df = df.withColumnRenamed(cfg["label"], "label")
        vector = VectorAssembler(inputCols=cfg["features"], outputCol="features") \
            .setHandleInvalid("skip").transform(df).select("features", "label")
        vector = vector.cache()
        n_rows = vector.count()
        out["n_rows"] = n_rows
        out["prepare_s"] = round(time.perf_counter() - t0, 1)
        model = SparkXGBRegressor(
            num_workers=cfg["num_workers"],
            objective="reg:squarederror",
            max_depth=cfg["max_depth"],
            num_boost_round=cfg["rounds"],
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=20,
            reg_lambda=2.0,
            tree_method="hist",
            max_bin=128,
            features_col="features",
            label_col="label",
            prediction_col="prediction",
            verbose_eval=False,
        )
        t1 = time.perf_counter()
        model.fit(vector)
        fit_s = time.perf_counter() - t1
        out.update(status="ok", fit_s=round(fit_s, 1),
                   s_per_round=round(fit_s / cfg["rounds"], 3))
    except BaseException as exc:
        lines = traceback.format_exception_only(type(exc), exc)[-1].strip().splitlines()
        out.update(status="fail", error_type=type(exc).__name__,
                   error_lines=[l.strip() for l in lines if l.strip()][:4])
    finally:
        spark.stop()
    print("PROBE_JSON " + json.dumps(out))

main()
"""


def run_config(master: str, workers: int, rows: int, rounds: int, depth: int,
               sample_rows: int, log) -> dict:
    import subprocess
    import tempfile

    script = Path(tempfile.gettempdir()) / "xgb_cost_calibration_sparkxgb_child.py"
    script.write_text(CHILD, encoding="utf-8")
    conf = dict(SPARK_CONF)
    conf["spark.local.dir"] = str(SPARK_LOCAL_DIR)
    conf["spark.sql.shuffle.partitions"] = "120"
    conf["spark.default.parallelism"] = "120"
    fraction = min(1.0, rows / sample_rows)
    cfg = {
        "master": master, "task_cpus": 1, "num_workers": workers, "rows": rows,
        "rounds": rounds, "max_depth": depth, "seed": SEED, "fraction": round(fraction, 6),
        "sample_dir": str(SAMPLE_DIR), "features": FEATURES, "label": LABEL, "conf": conf,
    }
    env = dict(os.environ)
    env["PYSPARK_PYTHON"] = sys.executable
    env["PYSPARK_DRIVER_PYTHON"] = sys.executable
    label = f"{master}/workers={workers}/rows={rows}"
    start = time.perf_counter()
    completed = subprocess.run([sys.executable, str(script), json.dumps(cfg)],
                               capture_output=True, text=True, env=env, timeout=7200)
    payload = [line for line in completed.stdout.splitlines() if line.startswith("PROBE_JSON ")]
    if payload:
        result = json.loads(payload[-1][len("PROBE_JSON "):])
    else:
        tail = (completed.stderr or completed.stdout).strip().splitlines()[-4:]
        result = {"status": "fail", "error_type": "no-json", "num_workers": workers,
                  "master": master, "rows": rows,
                  "error_lines": [line.strip() for line in tail]}
    result["wall_s"] = round(time.perf_counter() - start, 1)
    result["label"] = label
    if result.get("status") == "ok":
        log(f"[{label}] fit {result['fit_s']} s on {result['n_rows']:,} rows "
            f"({result['s_per_round']} s/round)")
    else:
        log(f"[{label}] FAILED: {(result.get('error_lines') or [''])[0][:160]}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, master_default="local[8]", shuffle_default=120)
    parser.add_argument("--rows", type=int, default=2_000_000,
                        help="training rows per config (default 2 M)")
    parser.add_argument("--workers", default="4,8",
                        help="comma separated num_workers values (default 4,8)")
    parser.add_argument("--rounds", type=int, default=10, help="boosting rounds (default 10)")
    parser.add_argument("--max-depth", type=int, default=6, help="tree depth (default 6)")
    args = parser.parse_args(argv)

    log = open_tee("14_spark_xgb_probe")
    ensure_dir(SPARK_LOCAL_DIR)
    sample_rows = 23_590_754  # rows written by 10_scan_and_sample.py at 5 %
    result: dict = {"stage": "spark_xgb_probe", "master": args.master,
                    "rows_target": args.rows, "rounds": args.rounds,
                    "max_depth": args.max_depth, "runs": []}

    for workers in [int(w) for w in args.workers.split(",") if w.strip()]:
        result["runs"].append(run_config(args.master, workers, args.rows, args.rounds,
                                         args.max_depth, sample_rows, log))

    ok = [run for run in result["runs"] if run.get("status") == "ok"]
    if len(ok) >= 2:
        best = min(ok, key=lambda run: run["s_per_round"])
        worst = max(ok, key=lambda run: run["s_per_round"])
        result["speedup_best_vs_slowest"] = round(worst["s_per_round"] / best["s_per_round"], 3)
        result["fastest"] = best["label"]
    result["env"] = env_meta()
    log(f"summary -> {write_metrics('spark_xgb_probe', result)}")
    log.close()
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
