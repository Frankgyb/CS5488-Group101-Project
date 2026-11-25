# XGBoost cost calibration (probes, not a model)

**Status: measurement only — no model is trained here, and nothing in this folder is a
deliverable ML result.** The supervised stage was decided against on 2026-11-25 (see
the report's future-work section); these probes exist so the numbers quoted there
can be regenerated on this machine instead of being remembered.

## What is here

| Script | Question it answers | Output |
| --- | --- | --- |
| `calibration_common.py` | shared paths, 26-feature whitelist, v3 Spark config | — |
| `10_scan_and_sample.py` | how long a full 471 M-row feature pass takes; cost of writing the 5 % sample | `metrics/scan_and_sample.json` |
| `11_xgb_train_probe.py` | driver-side XGBoost throughput and `nthread` scaling | `metrics/xgb_train_probe.json` |
| `12_spark_ml_probe.py` | JVM-side `GBT`/`RF` unit cost (no Python workers involved) | `metrics/spark_ml_probe.json` |
| `13_python_worker_probe.py` | why Python workers failed before, and what actually fixed them | `metrics/python_worker_probe.json` |
| `14_spark_xgb_probe.py` | unit cost of distributed `SparkXGBRegressor` (route P1) | `metrics/spark_xgb_probe.json` |
| `run_calibration.py` | runs 10→14 and digests everything | `metrics/calibration_summary.json` |

## Run it

From any directory — every path is resolved from this file, never from the CWD:

```bash
D:/Anaconda/python.exe project/code/ml/experiments/xgb_cost_calibration/run_calibration.py
D:/Anaconda/python.exe project/code/ml/experiments/xgb_cost_calibration/run_calibration.py --steps 11,12
```

Individual probes take the same flags (`--rows`, `--master`, `--threads`, `--workers`,
`--rounds`, `--sample-fraction`, `--skip-write`); `python <script> --help` lists them.

## Where the outputs go

```
project/data/output/xgb_cost_calibration/
├── metrics/            small JSON files — these are the evidence, keep them with the repo
├── sample5pct/         23.6 M-row Parquet sample, 1.36 GB — regenerable, do NOT commit
└── _spark_tmp/         Spark scratch dir, safe to delete
project/data/log/xgb_cost_calibration/    per-run logs (tee'd copies of stdout)
```

## Reading the numbers

* Every figure is a **single run**. Identical configurations on this laptop differ by up to
  26–42 % (see `data/output/scaling_study/README.md`), so use these to price work, not to claim
  speedups. Anything that must be quoted as a result needs 3+ repeats and a median.
* Both 3.7 s (three-row Spark count) and 38 s (5 % sample write) come from the same cold
  page cache; the scan numbers are I/O bound on Parquet metadata, not CPU bound.
* `11_...` and `14_...` measure different things: driver-side XGBoost trains in-process on
  one machine's cores, `SparkXGBRegressor` trains one model per worker and needs
  `spark.task.cpus` fixed at session start.

## The interpreter trap (read before "fixing" Spark)

`PYSPARK_PYTHON` must point at the same interpreter as the driver
(`D:/Anaconda/python.exe`). Without it Spark launches workers with `python3`, which does not
exist on this Windows install (Microsoft Store stub, resolves to nothing → `Timed out while
waiting for the Python worker to connect back`). `13_python_worker_probe.py` records both
sides of that: unset → fail, set → ok, plus the harmless `pyspark.daemon` ImportError
(`signal.SIGHUP` does not exist on Windows) that was originally mistaken for the cause.
