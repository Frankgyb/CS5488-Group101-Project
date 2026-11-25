# Location: project/code/ml/experiments/xgb_cost_calibration/13_python_worker_probe.py | PROJECT_ROOT depth: 4
"""Probe D: why Python workers failed, and whether they still do.

The first diagnosis of the blockage was wrong. Importing ``pyspark.daemon`` on this
machine does fail (``ImportError: cannot import name 'SIGHUP' from 'signal'``), but that is
*not* what killed the workers: with ``PYSPARK_PYTHON`` pointing at the Anaconda 3.14
interpreter, RDD jobs and ``SparkXGBRegressor`` both run. The real failure was interpreter
resolution — Spark launches workers with ``python3`` unless ``PYSPARK_PYTHON`` is set, and on
this machine ``python3`` is the Microsoft Store stub (exits immediately, so PySpark reports
``Timed out while waiting for the Python worker to connect back``), while ``python`` on PATH
is a Python 3.11 that carries a *different* PySpark (3.5.5) than the 4.2.1 driver.

This probe records all of that so the finding is reproducible rather than remembered:

  1. resolver  — what ``python3`` / ``python`` are and which PySpark they import,
  2. worker    — RDD ``map`` without ``PYSPARK_PYTHON`` (expected to FAIL),
  3. worker    — RDD ``map`` with ``PYSPARK_PYTHON=sys.executable`` (expected to SUCCEED),
  4. xgboost   — tiny ``SparkXGBRegressor`` fit with ``PYSPARK_PYTHON`` set (expected to
                 SUCCEED, which is what makes route P1 available after all),
  5. daemon    — ``import pyspark.daemon`` in-process (expected to FAIL, harmless).

Each Spark check runs in its own child process because the environment variable has to be
set before the session starts; nothing here needs the v3 tuning, so children use a minimal
session. Writes ``metrics/python_worker_probe.json``.

Usage:
    python project/code/ml/experiments/xgb_cost_calibration/13_python_worker_probe.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3].parent
sys.path.insert(0, str(HERE))

from calibration_common import env_meta, open_tee, write_metrics  # noqa: E402

# A child that starts a minimal session and reports one worker-path result as JSON.
CHILD = r"""
import json, sys, time, traceback
from pyspark.sql import SparkSession
mode = sys.argv[1]
rows = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
spark = (SparkSession.builder.appName("worker-probe").master(
    "local[4]").config("spark.ui.showConsoleProgress", "false").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
out = {"mode": mode}
try:
    if mode == "rdd":
        t = time.perf_counter()
        total = spark.sparkContext.parallelize(range(1000), 4).map(lambda x: x * 2).sum()
        out.update(status="ok", detail=f"sum={total}", seconds=round(time.perf_counter() - t, 1))
    else:
        from pyspark.ml.feature import VectorAssembler
        from pyspark.sql import functions as F
        from xgboost.spark import SparkXGBRegressor
        frame = (spark.range(rows)
                 .withColumn("miles", F.rand(1) * 30)
                 .withColumn("fare", F.rand(2) * 100)
                 .withColumn("label", F.rand(3) * 0.3))
        frame = VectorAssembler(inputCols=["miles", "fare"], outputCol="features") \
            .transform(frame).select("features", "label")
        t = time.perf_counter()
        SparkXGBRegressor(num_workers=2, objective="reg:squarederror", max_depth=3,
                          num_boost_round=5, features_col="features", label_col="label",
                          verbose_eval=False).fit(frame)
        out.update(status="ok", detail=f"fitted {rows} rows", seconds=round(time.perf_counter() - t, 1))
except BaseException as exc:
    lines = traceback.format_exception_only(type(exc), exc)[-1].strip().splitlines()
    out.update(status="fail", error_type=type(exc).__name__,
               error_lines=[l.strip() for l in lines if l.strip()][:4])
finally:
    spark.stop()
print("PROBE_JSON " + json.dumps(out))
"""

PROBE_SCRIPT = Path(tempfile.gettempdir()) / "xgb_cost_calibration_worker_child.py"


def run_child(mode: str, rows: int, python_env: str | None, log) -> dict:
    """Run one Spark check in a child process with (or without) PYSPARK_PYTHON."""
    PROBE_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    PROBE_SCRIPT.write_text(CHILD, encoding="utf-8")
    env = dict(os.environ)
    for key in ("PYSPARK_PYTHON", "PYSPARK_DRIVER_PYTHON"):
        env.pop(key, None)
    if python_env:
        env["PYSPARK_PYTHON"] = python_env
        env["PYSPARK_DRIVER_PYTHON"] = python_env
    label = f"{mode}/{'explicit' if python_env else 'unset'}"
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, str(PROBE_SCRIPT), mode, str(rows)],
            capture_output=True, text=True, env=env, timeout=900,
        )
        payload = [line for line in completed.stdout.splitlines()
                   if line.startswith("PROBE_JSON ")]
        if payload:
            result = json.loads(payload[-1][len("PROBE_JSON "):])
        else:
            tail = (completed.stderr or completed.stdout).strip().splitlines()[-4:]
            result = {"status": "fail", "error_type": "no-json",
                      "error_lines": [line.strip() for line in tail]}
    except subprocess.TimeoutExpired:
        result = {"status": "fail", "error_type": "timeout", "error_lines": ["child timed out"]}
    result["seconds"] = round(time.perf_counter() - start, 1)
    result["label"] = label
    log(f"[{label}] {result['status']} in {result['seconds']} s "
        f"{(result.get('detail') or (result.get('error_lines') or [''])[0])[:120]}")
    return result


def describe_interpreter(name: str, log) -> dict:
    """Resolve one interpreter name and report its version and PySpark version."""
    try:
        completed = subprocess.run(
            [name, "-c", "import sys;print(sys.version.split()[0]);"
                         "import importlib.util as u;"
                         "print('pyspark' if u.find_spec('pyspark') else 'no-pyspark');"
                         "import pyspark;print(pyspark.__version__) if u.find_spec('pyspark') else None"],
            capture_output=True, text=True, timeout=120,
        )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        info = {"name": name, "returncode": completed.returncode,
                "python": lines[0] if lines else None,
                "pyspark_installed": (lines[1] == "pyspark") if len(lines) > 1 else None,
                "pyspark": lines[2] if len(lines) > 2 else None,
                "stderr": (completed.stderr or "").strip().splitlines()[-2:]}
    except BaseException as exc:  # noqa: BLE001 - record whatever the OS says
        info = {"name": name, "returncode": None, "error": f"{type(exc).__name__}: {exc}"}
    log(f"[resolver:{name}] {json.dumps(info)}")
    return info


def daemon_import() -> dict:
    start = time.perf_counter()
    try:
        from pyspark import daemon  # noqa: F401

        return {"status": "ok", "detail": daemon.__file__,
                "seconds": round(time.perf_counter() - start, 1)}
    except BaseException as exc:  # noqa: BLE001
        lines = traceback.format_exception_only(type(exc), exc)[-1].strip().splitlines()
        return {"status": "fail", "error_type": type(exc).__name__,
                "error_lines": [line.strip() for line in lines if line.strip()][:4],
                "seconds": round(time.perf_counter() - start, 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000,
                        help="rows for the SparkXGB fit (default 100k)")
    parser.add_argument("--skip-xgb", action="store_true", help="skip the SparkXGB check")
    args = parser.parse_args(argv)

    log = open_tee("13_python_worker_probe")
    result: dict = {"stage": "python_worker_probe"}

    result["resolver"] = {
        "python3": describe_interpreter("python3", log),
        "python": describe_interpreter("python", log),
        "driver": sys.executable,
    }

    result["checks"] = {
        "rdd_map_unset": run_child("rdd", 1000, None, log),
        "rdd_map_explicit": run_child("rdd", 1000, sys.executable, log),
    }
    if not args.skip_xgb:
        result["checks"]["spark_xgb_explicit"] = run_child(
            "xgb", args.rows, sys.executable, log)
    result["checks"]["daemon_import"] = daemon_import()
    log(f"[daemon_import] {result['checks']['daemon_import']['status']} "
        f"{(result['checks']['daemon_import'].get('error_lines') or ['ok'])[0]}")

    checks = result["checks"]
    result["verdict"] = {
        "worker_blocked_without_env": checks["rdd_map_unset"]["status"] == "fail",
        "worker_ok_with_env": checks["rdd_map_explicit"]["status"] == "ok",
        "distributed_xgb_available": checks.get("spark_xgb_explicit", {}).get("status") == "ok",
        "root_cause": ("python interpreter resolution: without PYSPARK_PYTHON, Spark launches "
                       "workers with `python3`, which is not an executable here (returncode 9009)"),
        "daemon_import_failure_is_harmless": checks["daemon_import"]["status"] == "fail",
    }
    result["env"] = env_meta()
    log(f"verdict={json.dumps(result['verdict'])}")
    log(f"metrics -> {write_metrics('python_worker_probe', result)}")
    log.close()
    return 0 if (result["verdict"]["worker_ok_with_env"]
                 and result["verdict"]["worker_blocked_without_env"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
