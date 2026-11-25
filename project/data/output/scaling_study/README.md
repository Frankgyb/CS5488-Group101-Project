# scaling_study — v3 core-count / `n_jobs` evidence

This directory is the **reproducible evidence bundle** for the project's
parallel-computing section. It was produced by the code in
`project/code/ml/experiments/parallel_scaling/` (see its `README.md`).

**Scope:** a 2,000,000-row K1/K2 study and a 1,000,000-row IF scoring study,
sampled from `data/output/sampled_km` (46,706,095 rows, 16 parts, seed 42).
It is **not** a 471 M-row production speed-up measurement.

## Headline results

| Stage | Recommendation | Evidence |
|---|---|---|
| `KMeans(k=2)` @ 2 M | **`local[4]`** | 1.44–1.63× vs `local[2]`; `local[8]` flat/slower |
| `BisectingKMeans(k=8)` @ 2 M | **`local[8]`** | 1.79–1.89× vs `local[2]` (primary n=4); S=30 post-hoc pooled 1.77× |
| IF `decision_function` @ 1 M | **`n_jobs=8`**, not `-1` | 4.38× vs 1 thread; 20 threads 2.12 s vs 1.94 s |

K1 4→8 is the weakest claim: S=30 is noisy, S=120 is stable and slightly slower
at 8. Say "no evidence that 8 beats 4", not "8 is identical to 4". For K2 the
primary 4-run/cell estimate is **1.79× (S=30, CI 1.23–2.22)** and **1.89×
(S=120, CI 1.85–2.20)**. The S=30 8-core cell has one 143.8 s outlier, so quote
its CI, not just the median.

A post-hoc replication of the S=30 local[8] cell (4 fresh runs, after observing
the 143.8 s point) did **not** reproduce that outlier (max/median 1.15), but its
median was 108.4 s (10.1% slower than primary). Pooling the 8 runs gives
**1.77× [1.41, 2.20]**; this is an unbalanced sensitivity, not a replacement.
So K2 is "about 1.8×" with S=120 as the more stable cell; do not claim a stable
2.2×.

## Files

| File | Meaning |
|---|---|
| `s2_k1.csv` / `s2_k1_runs.jsonl` / `s2_k1_summary.csv` | per-run, raw JSONL, and per-cell summary for K1 |
| `s2_k2.csv` / `s2_k2_runs.jsonl` / `s2_k2_summary.csv` | same for K2 (n=4) |
| `s3_f1.csv` / `s3_f1_runs.jsonl` / `s3_f1_summary.csv` | same for IF scoring |
| `bootstrap_ci.json` | 20k-resample descriptive CIs for cell medians and speedups |
| `checks.json` | completeness / row-count / score-equality checks; `pass: true` |
| `env.json` | machine, Python/PySpark/sklearn versions, power/space baseline |
| `run_status.json` | the `--analysis-only` regeneration step log |
| `MANIFEST.json` | SHA-256 and byte size of every file in this directory |
| `SUMMARY.md` | generated tables + automatic checks |
| `figures/` | `s2_speedup.png`, `s2_parallel_eff.png`, `s3_speedup.png`, `s3_parallel_eff.png` |
| `replication/k2_s30_c8/` | post-hoc S=30 local[8] raw CSV/JSONL (4 fresh runs) |
| `replication/analysis/` | replication summary CSV/JSON, Markdown note, figure |
| `smoke/` | smoke-test CSV/summary from the full harness |
| `subsets/` | materialised 2 M subset; the 1 M subset is regenerated on demand |

## Reproduce

```bash
# full study (rebuilds subsets, runs Spark, writes everything)
python project/code/ml/experiments/parallel_scaling/run_scaling_study.py

# K2-only rerun: 6 cells x (1 warm-up + 4 measured), formal layout
python project/code/ml/experiments/parallel_scaling/run_scaling_study.py --only k2 --repeats-k2 5 --k2-order random --sandbox-temp --force

# only regenerate figures/checks/bootstrap from the CSVs already here
python project/code/ml/experiments/parallel_scaling/run_scaling_study.py --analysis-only

# post-hoc K2 S=30 local[8] replication analysis (reads CSV/JSONL only)
python project/code/ml/experiments/parallel_scaling/analyze_k2_replication.py
```

CLI relative paths are resolved against the `project/` directory, not the shell
cwd. From the repository root either omit the leading `project/` (e.g.
`data/output/...`) or pass an absolute path.

In the DSH Windows sandbox add `--sandbox-temp` if the JVM gateway cannot create
its temp directory.

## Important caveats

- Whole-machine CPU%, not per-core; `parallel_eff` is a diagnostic, not speed-up.
- No temperature telemetry; the noisy `local[8]` S=30 cell is described, not
  fully explained. K2 S=30 8-core has one 143.8 s outlier; its CI lower bound
  (1.23) is below 1.5, so the 8-core benefit over 2 is positive (the interval
  excludes 1) but the interval is too wide to claim a >1.5× gain. K2 S=120
  is stable (1.89×, CI 1.85–2.20). The post-hoc S=30 replication did not
  reproduce 143.8 s, but its median was 10.1% slower; quote the pooled 1.77×
  only as a sensitivity, never as a stable 2.2×.
- `local[1]` is a P/E-core scheduling reference only and is not the main
  denominator.
- The full-data `BisectingKMeans(k=8)` 2,198 s timing and the old sub-5-second
  benchmarks remain **invalid as scaling evidence**; see
  `handoff/ML/timing_provenance.md` and `handoff/ML/parallel_scaling_v3.md`.

The detailed narrative, tables, CIs, limitations and report-ready wording are in
`handoff/ML/parallel_scaling_v3.md`.
