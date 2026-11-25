# Location: project/code/ml/experiments/xgb_cost_calibration/run_calibration.py | PROJECT_ROOT depth: 4
"""One entry point for the four XGBoost cost-calibration probes.

Runs, in order:
  10_scan_and_sample.py      Spark feature-pass + 5 % sample cost
  11_xgb_train_probe.py      driver-side XGBoost throughput and nthread scaling
  12_spark_ml_probe.py       JVM-side GBT / RF unit cost
  13_python_worker_probe.py  capability check (expected to report the Windows limitation)

Each step is a separate process, because Spark sessions cannot reconfigure ``master`` or
``task.cpus`` at runtime. A step that fails is logged and the remaining steps still run;
step 13 is expected to "fail" its SparkXGB check and exits 3 when the environment still
matches the documented limitation.

Writes ``metrics/calibration_summary.json`` (a compact digest of all four metric files) and
``logs/run_calibration.log``.

Usage:
    python project/code/ml/experiments/xgb_cost_calibration/run_calibration.py
    ... --steps 11,12          # only some steps
    ... --skip 13              # everything but the capability check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration_common import (  # noqa: E402
    HERE, METRICS_DIR, env_meta, open_tee, write_metrics,
)

STEPS = {
    "10": "10_scan_and_sample.py",
    "11": "11_xgb_train_probe.py",
    "12": "12_spark_ml_probe.py",
    "13": "13_python_worker_probe.py",
    "14": "14_spark_xgb_probe.py",
}
METRIC_FILES = {
    "10": "scan_and_sample",
    "11": "xgb_train_probe",
    "12": "spark_ml_probe",
    "13": "python_worker_probe",
}


def digest(name: str) -> dict:
    path = METRICS_DIR / f"{name}.json"
    if not path.is_file():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def summarise() -> dict:
    scan = digest("scan_and_sample")
    train = digest("xgb_train_probe")
    spark_ml = digest("spark_ml_probe")
    worker = digest("python_worker_probe")
    dist = digest("spark_xgb_probe")
    summary = {
        "stage": "calibration_summary",
        "env": env_meta(),
        "feature_pass": {
            "all_months_s": scan.get("all_months_s"),
            "all_months_rows": scan.get("all_months_rows"),
            "two_months_s": scan.get("two_months_s"),
            "sample_write_s": scan.get("sample_write_s"),
            "sample_rows": scan.get("sample_rows"),
            "sample_gb": scan.get("sample_gb"),
            "sample_zero_tip_share": scan.get("sample_zero_tip_share"),
        },
        "driver_xgb": {
            "n_rows": train.get("n_rows"),
            "matrix_gb": train.get("matrix_gb"),
            "dmatrix_build_s": train.get("dmatrix_build_s"),
            "depth6": train.get("depth6"),
            "depth8": train.get("depth8"),
            "thread_scaling": train.get("thread_scaling"),
        },
        "jvm_spark_ml": {
            "n_rows": spark_ml.get("n_rows"),
            "read_assemble_s": spark_ml.get("read_assemble_s"),
            "gbt": spark_ml.get("gbt"),
            "rf": spark_ml.get("rf"),
        },
        "distributed_xgb": {
            "rows_target": dist.get("rows_target"),
            "rounds": dist.get("rounds"),
            "runs": [{k: run.get(k) for k in ("label", "status", "n_rows", "fit_s",
                                              "s_per_round")}
                     for run in dist.get("runs", [])],
            "speedup_best_vs_slowest": dist.get("speedup_best_vs_slowest"),
        },
        "python_worker": {
            "verdict": worker.get("verdict"),
            "resolver": {k: {"python": v.get("python"), "pyspark": v.get("pyspark"),
                             "returncode": v.get("returncode")}
                         for k, v in (worker.get("resolver") or {}).items()
                         if isinstance(v, dict)},
            "daemon_import_error": (worker.get("checks", {})
                                    .get("daemon_import", {})
                                    .get("error_lines")),
        },
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", default="10,11,12,13,14",
                        help="comma separated step ids (default 10,11,12,13,14)")
    parser.add_argument("--skip", default="", help="comma separated step ids to skip")
    parser.add_argument("--summary-only", action="store_true",
                        help="do not run anything, only rebuild the summary from metrics")
    parser.add_argument("--forward-args", default="",
                        help="extra arguments appended to every step (quoted string)")
    args = parser.parse_args(argv)

    log = open_tee("run_calibration")
    chosen = [s.strip() for s in args.steps.split(",") if s.strip()]
    skipped = {s.strip() for s in args.skip.split(",") if s.strip()}
    if args.skip == "all":
        chosen = []
    forward = args.forward_args.split() if args.forward_args else []

    results: dict[str, dict] = {}
    for step in chosen:
        if step in skipped:
            log(f"step {step}: skipped")
            results[step] = {"status": "skipped"}
            continue
        script = STEPS.get(step)
        if script is None:
            log(f"step {step}: unknown, ignored")
            continue
        command = [sys.executable, str(HERE / script), *forward]
        log(f"step {step}: {' '.join(command)}")
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=str(HERE))
        seconds = round(time.perf_counter() - start, 1)
        status = "ok" if completed.returncode == 0 else f"exit {completed.returncode}"
        results[step] = {"status": status, "seconds": seconds,
                         "exit_code": completed.returncode}
        log(f"step {step}: {status} in {seconds} s")

    summary = summarise()
    summary["steps"] = results
    summary["note"] = (
        "Single-run calibration, not a controlled benchmark: per timing_provenance.md, "
        "identical configurations can differ by 26-42 %, so these numbers price the work "
        "and must not be quoted as deliverable results."
    )
    path = write_metrics("calibration_summary", summary)
    log(f"summary -> {path}")
    log(json.dumps(summary.get("driver_xgb", {}), indent=2))
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
