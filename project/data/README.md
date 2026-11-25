# project/data — what is tracked and what is regenerated

This directory holds the data layer of the project. Only the **small artefacts that back the
numbers in the report** are tracked; the raw release and every bulk intermediate are excluded
from version control and rebuilt with the scripts in `../code/`.

## Tracked

| Path | What it is |
|---|---|
| `log/` | Run logs for every stage. `log/out_of_use/README.md` maps superseded logs to the runs that produced them. |
| `output/if_results_rowkeyed/` | `run_meta.json`, `shap_rowkeyed.csv` and the 13,577-row `strict_anomalies_rowkeyed/` table (the 172 MB full-score table is excluded). |
| `output/clustering_robustness/` | Authoritative K sweep, silhouette curve, bootstrap ARI, `row_uid` cluster labels. |
| `output/anomaly_features/`, `output/anomaly_features_v2/` | Cluster tables named by cluster, `cluster_money_summary.csv`, report figures, unit-corrected enriched features. |
| `output/anomaly_clusters_rowkeyed/` | Cluster partition and the historical K=4 self-check (`selfcheck_k4/`, not citable). |
| `output/shap_final/` | Corrected SHAP figures, `shap_top_drivers_final.csv`, `discrimination_test.txt`. |
| `output/graph/` | Descriptive EDA figures. |
| `output/k_selection_diagnostics/` | Side-by-side evidence for the two silhouette scales (see its `README.md`). |
| `output/unit_check/`, `output/if_repro_probe/` | Unit and reproducibility probes. |
| `output/scaling_study/` | v3 parallel-scaling evidence bundle: raw per-run CSV/JSONL, summaries, `checks.json`, `MANIFEST.json`, four figures, and the 2 M-row subset. |
| `output/scaling_study_archive/` | Older n=2 evidence, kept as history only; not cited. |
| `output/xgb_cost_calibration/` | Cost-calibration `metrics/*.json` and its `README.md` (the 1.3 GB `sample5pct/` input is excluded). |
| `output/out_of_use/README.md` | Audit of the superseded artefacts that are **not** tracked. |

Hadoop `.crc` checksum files are not tracked (they are machine-local); Spark `_SUCCESS`
markers are kept.

## Not tracked

| Path | Size | Rebuild with |
|---|---|---|
| `parquet/` | 11.6 GB | `code/download.py` (24 monthly TLC files, 2023-01 … 2024-12) |
| `output/cleaned/` | 16 GB | `code/preprocessing/preprocessing.py` |
| `output/out_of_use/_outputs/` | 1.4 GB | Superseded outputs; documented in `output/out_of_use/README.md` |
| `output/sampled_km/` | 800 MB | `code/ml/pipeline/01_sample_v2.py` |
| `output/kmeans_full_result/` | 817 MB | `code/ml/experiments/kmeans_full.py` |
| `output/bi_aggregates_compact/` | 558 MB | `code/preprocessing/bi_aggregation.py` |
| `output/if_results_rowkeyed/full_scores_rowkeyed/` | 172 MB | `code/ml/pipeline/02b_isolation_forest_shap_rowkeyed.py` |
| `output/xgb_cost_calibration/sample5pct/` | 1.3 GB | `code/ml/experiments/xgb_cost_calibration/10_scan_and_sample.py` (≈31–37 s) |

Section 3.1 of [`../README.md`](../README.md) gives the same table with the partitioning and
file-count details used in the report.
