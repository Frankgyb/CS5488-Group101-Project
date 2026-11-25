# Parallel scaling study (v3)

This package contains the **single-machine core-count / `n_jobs` study** for the
project. It is the formal, project-relative version of the scratch experiment
that produced the numbers in `../../../../data/output/scaling_study/` and
summarised in the project report (Part III).

It does **not** change the production pipeline, the 13,577 anomaly flags, the
K=2 clustering result, or any existing output.

## What is measured

| Experiment | Model | Rows | Axis | Repeats |
|---|---|---|---|---|
| K1 | `KMeans(k=2, seed=42, maxIter=20, tol=1e-4)` | 2,000,000 | `local[2,4,8]` × shuffle 30/120 | 1 warm-up + 4 measured |
| K2 | `BisectingKMeans(k=8, seed=42, maxIter=20)` | 2,000,000 | `local[2,4,8]` × shuffle 30/120 | 1 warm-up + 4 measured |
| S3/F1 | `IsolationForest(150, 512, 0.8, seed=42).decision_function` | 1,000,000 | `n_jobs {1,2,4,8,-1}` | 1 warm-up + 4 measured |

For K1/K2, only `fit()` is timed; `VectorAssembler` + `StandardScaler` +
`cache()` + `count()` happen once per cell and are reported separately. Every
cell uses a fresh JVM/SparkSession. For S3, the forest is fitted once (on the
driver) and only `decision_function` is timed; scores are checked against the
`n_jobs=1` reference (`max_abs_diff < 1e-10`).

`parallel_eff = (cpu_pct_mean / 100) × logical_cores / requested_units` is a
diagnostic, not a speed-up. It uses whole-machine `psutil.cpu_percent` at
0.1 s and includes the driver, GC and idle phases. `parallel_eff_net` subtracts
the per-cell idle baseline; it is often close to zero when the baseline is of
the same order as the Spark signal.

## Data

Input is the existing 10 % stratified sample:

```
project/data/output/sampled_km/     # 46,706,095 rows, 16 part-*.parquet
```

The 2 M and 1 M subsets are drawn deterministically with
`sample(n=rows, random_state=42)`. The 2 M subset is retained under
`project/data/output/scaling_study/subsets/subset_2m/`; the 1 M subset was used
for the S3 scoring run and is not kept (regenerate it with
`prepare_subset.py --from-subset ...`). Both are reproducible and are not
required once the raw CSVs are present: `--analysis-only` regenerates the
figures and checks from those CSVs.

## Run

From the repository root (or any cwd; all paths are resolved from the file
location):

```bash
# full study: subsets -> smoke -> K1/K2 -> S3 -> plots -> bootstrap
python project/code/ml/experiments/parallel_scaling/run_scaling_study.py

# same, including the optional local[1] reference point
python project/code/ml/experiments/parallel_scaling/run_scaling_study.py --include-local1

# regenerate figures/checks/bootstrap from already-persisted CSVs (no Spark)
python project/code/ml/experiments/parallel_scaling/run_scaling_study.py --analysis-only

# K2-only rerun, with random grid order to reduce heat/order drift
python project/code/ml/experiments/parallel_scaling/run_scaling_study.py --only k2 --repeats-k2 5 --k2-order random --sandbox-temp --force

# post-hoc S=30 local[8] replication analysis (reads persisted CSV/JSONL; no Spark)
python project/code/ml/experiments/parallel_scaling/analyze_k2_replication.py
```

CLI relative paths are resolved against the `project/` directory, not the shell
cwd. From the repository root either omit the leading `project/` or pass an
absolute path.

On a normal desktop, no extra flags are needed. In the DSH Windows sandbox the
PySpark gateway can fail when it cannot create its own temp directory; then add
`--sandbox-temp` (or set `SCALING_SANDBOX_TEMP=1`), which redirects the JVM
gateway/base temp to `data/output/scaling_study/_spark_gateway/`. The study also
uses `spark.local.dir=data/output/scaling_study/_spark_tmp/` rather than the
old hard-coded `D:/temp_spark`.

`run_scaling_study.py` checks AC power, free space, existing Java processes and
a 5 s idle CPU baseline before starting. It writes `env.json`, `run_status.json`,
per-step logs under `data/log/scaling_study/`, and a `MANIFEST.json` of output
hashes. A failed step writes `FAILURE.md` and stops; completed JSONL rows are
kept, so a rerun with `--resume` skips finished cells.

## Outputs

`project/data/output/scaling_study/`:

```
s2_k1.csv, s2_k1_runs.jsonl, s2_k1_summary.csv
s2_k2.csv, s2_k2_runs.jsonl, s2_k2_summary.csv
s3_f1.csv, s3_f1_runs.jsonl, s3_f1_summary.csv
bootstrap_ci.json
checks.json, env.json, run_status.json, MANIFEST.json
SUMMARY.md
figures/s2_speedup.png, s2_parallel_eff.png,
        s3_speedup.png, s3_parallel_eff.png
replication/k2_s30_c8/       # post-hoc raw CSV/JSONL for this one cell
replication/analysis/        # replication summary CSV/JSON, .md, figure
smoke/            # smoke CSV/summary from the full harness
subsets/          # created by a full run; optional for --analysis-only
```

The summary CSV columns are the ones used in the report: `n_cores`,
`shuffle_partitions`, `n_runs`, `fit_s_median`, `fit_s_mad`,
`speedup_vs_baseline`, `efficiency`, `parallel_eff_median`, Amdahl
`amdahl_alpha` / `amdahl_r2`（2/4/8 主拟合），以及
`amdahl_alpha_with_local1` / `amdahl_r2_with_local1`（含 local[1] 的敏感性）；
S3 uses `score_s_median`, `speedup_vs_1`, `efficiency`,
`parallel_eff_median`, `max_abs_diff_max`.

## Reading order

1. the project report (Part III) — detailed design, results,
   uncertainty, limitations and report-ready wording.
2. `../../../../data/output/scaling_study/SUMMARY.md` — generated tables/checks.
3. This README — how to run and what each file means.

## Reproducibility notes

- The raw result CSVs/JSONL in `data/output/scaling_study/` are the validated
  v3 run; `--analysis-only` regenerated the figures/checks from them.
- A full re-run on a different machine is expected to produce different
  absolute times. Compare **within-machine ratios**, not absolute seconds.
- K1's 4→8 comparison is the weakest main claim: the S=30 8-core cell is noisy
  and S=120 shows 8 slightly slower. The safe wording is "no evidence that 8
  beats 4", not "8 is proven identical to 4".
- K2's primary 4-run/cell estimate is 1.79× (S=30) and 1.89× (S=120). The S=30
  8-core cell has one 143.8 s outlier; quote its CI.
- A post-hoc replication of that S=30 cell (4 fresh runs) did not reproduce
  143.8 s (max/median 1.15), but its median was 10.1% slower; pooled 8 runs give
  1.77× [1.41, 2.20]. This is a sensitivity, not a replacement for the primary
  n=4 table; do not claim a stable 2.2×.
- The 2 M/1 M subsets are not the 471 M-row production input. Do not quote
  these numbers as production speed-ups; a 10 M-row re-test is the next step.
