# `data/log/out_of_use/` — archived run logs (2026-11-22)

Logs are never edited or deleted; they are only **moved** here when the run they
describe no longer produces anything the report quotes. Active logs stay in
`data/log/`. The authoritative directory for every log is still `data/log/`
(one level up) — this sub-folder is part of it.

## What is in here and why

| File(s) | Produced by | Why archived |
|---|---|---|
| `if_10pct_of_10pct_20261119_025142.log`, `if_10pct_of_10pct_20261119_055410.log` | legacy `02_isolation_forest_shap.py` (now `code/ml/out_of_use/code_legacy_shap/`) | superseded by the row_uid-keyed rerun `02b_rowkeyed_20261122_005037.log` |
| `if_dynamic_*.log`, `if_full_dynamic_*.log`, `if_proto_A/B_*.log`, `if_stable_*.log` | early Isolation-Forest prototypes (full-data / dynamic-threshold / sklearn-vs-Spark) | exploration only; none of their numbers appear in the report. They are also the only place where the OLD non-rowkeyed anomaly table was built |
| `sample_log_v1_legacy.txt` | `code/test/ml_legacy/sample_v1_baseline.py` (archived script) | the live sampling log is `../sample_v2.txt` (`01_sample_v2.py`). The two files print the **same** three headline numbers (471,960,468 → 467,142,106 → 467,067,101), differing only in one label line ("after speed filter and feature engineering" vs "after feature engineering"), so the file name — not the content — is what distinguishes them. Renamed on archive to make the provenance explicit |
| `benchmark_demo_20261118_014502.txt` | `code/preprocessing/benchmark_demo.py` (one-month toy run) | exploratory only: every timing is 0.26–0.39 s, i.e. pure Spark start-up cost, and the demo never ran `local[8]` or the 2048 MB partition. Cited nowhere |
| `benchmark_full_20261118_014818.txt` | `code/preprocessing/benchmark.py`, earlier attempt | **aborted run** — it stops right after `Partition 32m: Discovered 24 partitions` with no timing (9 lines). Superseded by the complete run below; kept only as evidence that nothing was cherry-picked |
| `eda_log_20261117_225329.txt`, `full_eda_log_20261117_230904.txt` | full-dataset EDA (471.96 M rows read from `data/parquet`) | that input is no longer on disk (only `cleaned` / `sampled*` remain), so these are historical records. The live EDA log `eda_clustering.txt` is archived here too (see next row) |

| `eda_clustering.txt` | `analysis/09_eda_clustering.py` (run of 2026-11-19) | reads the OLD non-rowkeyed table (`if_results_10pct_of_10pct/strict_anomalies`) and stops mid-file at `=== Feature Skewness ===`, so it is neither current nor complete. Descriptive statistics quoted in the report come from `handoff/ML/handoff_file.md` §EDA, not from this log |

## Reports: which log to cite

| Number quoted in the report | Cite |
|---|---|
| anomaly count 13,577, IQR threshold −0.1664, 0.29 %, 4,672,734 sampled rows, 2,000 SHAP rows | `../02b_rowkeyed_20261122_005037.log` **and** `data/output/if_results_rowkeyed/run_meta.json` |
| sampling funnel (471.96 M → 467.07 M) | `../sample_v2.txt` |
| K selection (K=2, silhouette 0.2198 ± 0.0000, bootstrap ARI 0.96) | `data/output/clustering_robustness/k_sweep.csv`, `chosen_model.json`, `bootstrap_ari.csv` |
| partition / CPU / column-pruning benchmark (partitions 32/128/512/2048 MB) | `benchmark_full_20261118_015057.txt` (this folder) |
| per-cluster SHAP top-3, ρ = 0.9833 | `data/output/shap_final/shap_top_drivers_final.csv`, `discrimination_test.txt` |

⚠️ **Do not cite** `../kmeans_cluster_anomalies_output.txt`'s silhouette column
(0.3662 at K=2). It was produced from the OLD non-rowkeyed anomaly table and its
silhouette comes from Spark's squared-Euclidean scale (not inflated) — see
`handoff/ML/readme_Isolation_tree_Kmeans.md` §3.3.
The file stays in `data/log/` only because it is the evidence for that finding.

| `shap_cluster_analysis_v1_legacy.txt` | 旧 `07_shap_cluster_analysis.py`（按行位置贴簇标签，已作废） | 2026-11-22 从现役 `data/log/` 移入。日志里 `Row count mismatch. Truncating clustered data to 2000 rows.` 正是那个 bug 的现场；簇分布 1445/555 与现役 K=2 的 9791/3786 口径完全不同 | ❌ 两侧没有公共键，标签等于随机分配。逐簇 SHAP 一律用 `data/output/shap_final/shap_top_drivers_final.csv` |
