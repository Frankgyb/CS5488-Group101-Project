# CS5488 Group 101 — NYC FHV Trip Records

Parallel analytics on 471,960,468 NYC for-hire-vehicle trips: EDA, anomaly detection, and
behavioural clustering, all executed on a **single machine** with PySpark (`local[*]`).

This README is a short orientation to the deliverable. The full work logic and results are in
the project report (`../report/main.pdf`); the numbers can be traced through the per-directory
READMEs under `data/output/` and the evidence bundle in `data/output/scaling_study/`.

---

## 1. What the project answers

| # | Question | Method | Main output |
|---|---|---|---|
| a | How do demand, fare, and tipping distribute over time, space, and platform? | Spark SQL over Parquet, pre-aggregated to daily grain | `data/output/bi_aggregates_compact/` (delivered to a separate BI track; not analysed here) |
| b | Which trips are abnormal in fare, distance, duration, or tip? | IsolationForest (sklearn, fitted on driver) + SHAP attribution | `data/output/if_results_rowkeyed/`, `data/output/shap_final/` |
| c | Do those anomalies form interpretable behavioural groups? | KMeans, K chosen by multi-seed silhouette + bootstrap ARI | `data/output/clustering_robustness/`, `data/output/anomaly_features*/` |

The organising constraint is the one the assignment is built around: **a 11.6 GB / 471 M-row dataset
has to be processed end to end on one laptop-class machine**, so every stage is written as a Spark
job rather than a pandas script.

## 2. Results at a glance

| What | Result | Where |
|---|---|---|
| Trips cleaned | 471,960,468 raw → 467,142,106 after flag filtering (99.0%) | `data/log/sample_v2.txt` |
| Working sample | 46,706,095 trips (10%), stratified over 6 layers | `data/output/sampled_km/` |
| Anomalies flagged | **13,577** (0.291% of the scored 4,672,734), IQR threshold −0.1664 | `data/output/if_results_rowkeyed/run_meta.json` |
| Cluster count | **K = 2**, silhouette 0.2198, chosen by 10/10 seeds; bootstrap ARI 0.96 ± 0.03 scored on out-of-bag trips | `data/output/clustering_robustness/` |
| Cluster A — `short_high_tip` | 9,791 trips (72.1%): median fare $134.69, 34.4 mi, 59 min, tip rate 20.6% | `data/output/anomaly_features/cluster_money_summary.csv` |
| Cluster B — `long_low_tip` | 3,786 trips (27.9%): median fare $210.90, 62.5 mi, 88 min, **zero tip on 56.3%** | same |
| SHAP drivers | Both clusters rank `log_tips` > `speed_mph` > `log_fare`; Spearman ρ = 0.9833 → the two groups are separated by *magnitude*, not by which features matter | `data/output/shap_final/discrimination_test.txt` |

If only one number is quoted, it is **K = 2 with 13,577 anomalies falling into a short-trip/high-tip
group and a long-trip/low-tip group**.

## 3. Repository layout

```
project/
├── README.md                # this file
├── code/                    # 55 Python files (44 live, 7 legacy checks, 4 superseded)
│   ├── download.py
│   ├── preprocessing/       # cleaning, EDA, BI aggregation, benchmarks
│   ├── ml/pipeline/         # 01 sample → 02b IsolationForest+SHAP → 03/04 clustering → 06 figures
│   ├── ml/analysis/         # 05 money summary, 07b/10/11 clustering and SHAP, 08/09 figures, v2 consistency
│   ├── ml/experiments/      # K selection sweeps, full-data bisecting KMeans, parallel_scaling/,
│   │                        # xgb_cost_calibration/ (cost evidence for the future-work section)
│   ├── ml/tools/            # unit and reproducibility probes
│   ├── ml/out_of_use/       # superseded and legacy scripts (kept for provenance only)
│   └── test/                # validation scripts
└── data/
    ├── output/              # all derived artefacts
    └── log/                 # run logs; logs of superseded runs are under log/out_of_use/
```

Every script resolves the project root from its own location (`Path(__file__).resolve().parents[n]`)
and never from the current working directory, so scripts can be launched from anywhere.

### 3.1 Large data directories

The bulk data is **not** part of the submission. The directories below keep their shape
(partitioning scheme and column layout) so the pipeline can be recreated from `download.py`;
only the file counts and sizes are reported here.

| Directory | Size | Original structure |
|---|---|---|
| `data/parquet/` | 11.6 GB | 24 monthly TLC Parquet files (2023-01 … 2024-12), 300–522 MB each (11.3 GiB on disk) |
| `data/output/cleaned/` | 16 GB | Spark output partitioned by `year=/month=`, 75 data files (152 including Spark checksum files) |
| `data/output/out_of_use/` | 1.4 GB | Superseded outputs of earlier pipeline versions, **never cite**. Includes `_outputs/sampled_v1_superseded/` (1.3 GB, the v1 sample, 139 Parquet files, read by nothing in the live pipeline — do not confuse it with `sampled_km/`) |
| `data/output/sampled_km/` | 800 MB | First 10% stratified sample, output of `ml/pipeline/01_sample_v2.py`, 16 Parquet files; input of `02b` |
| `data/output/kmeans_full_result/` | 817 MB | BisectingKMeans K=8 fitted on the full 46.7 M sample, 16 Parquet files |
| `data/output/bi_aggregates_compact/` | 558 MB | Two pre-aggregated layers, each 731 daily partitions: `l1_detail/pickup_date=YYYY-MM-DD/` (14 columns, incl. `PULocationID` and platform) and `l2_overview/pickup_date=YYYY-MM-DD/` (zone-agnostic, 6,045 rows/day) |
| `data/output/if_results_rowkeyed/` | 173 MB | Spark output: `strict_anomalies_rowkeyed/` (13,577 rows) and `shap_rowkeyed/` (2,000 rows), plus `run_meta.json` |
| `data/output/xgb_cost_calibration/` | 1.4 GB total, **1.36 GB of it is `sample5pct/`** | Cost evidence behind the supervised-modelling future-work section: `metrics/*.json` (six small files: data-side scan, driver-side XGBoost, JVM GBT/RF, distributed `SparkXGBRegressor`, worker-capability, summary) produced by `ml/experiments/xgb_cost_calibration/`. Only the JSONs need shipping — `sample5pct/` (23.59 M rows) is regenerated in ~31–37 s by `10_scan_and_sample.py`, and `_spark_tmp/` is scratch |
| `data/output/scaling_study/` | 36 MB | v3 parallel-scaling evidence + the materialised 2 M subset (the subset alone is ~35 MB; the 1 M subset is not kept, regenerate it for a full S3 run): K1/K2/IF raw + summary CSV/JSONL, 4 PNGs, `checks.json`, `MANIFEST.json`, `bootstrap_ci.json`; code in `ml/experiments/parallel_scaling/`. The 2 M subset is reproducible and can be deleted; results without it are ~1 MB |

Everything else in `data/output/` is small and is included as-is. The directories that carry
report material:

| Directory | Produced by | Contents |
|---|---|---|
| `data/output/clustering_robustness/` | `ml/analysis/10_robust_clustering_rowkeyed.py` | Authoritative K: `k_sweep.csv`, `silhouette_curve.png`, `bootstrap_ari.csv`, `chosen_model.json`, `row_uid_cluster_robust.parquet` |
| `data/output/shap_final/` | `ml/analysis/11_final_shap_plots.py` | Corrected SHAP figures and `shap_top_drivers_final.csv`, `discrimination_test.txt` |
| `data/output/anomaly_features/` | `ml/pipeline/04_rename_clusters.py`, `06_plot_cluster_viz.py` | Cluster tables named by cluster, `cluster_money_summary.csv`, and `viz/` (four figures) |
| `data/output/anomaly_features_v2/` | `ml/analysis/add_consistency_features_v2.py` | Unit-corrected consistency metrics, `consistency_profile_v2.csv` |
| `data/output/anomaly_clusters_rowkeyed/` | `ml/pipeline/03_kmeans_cluster_anomalies.py` | Cluster partition only (`cluster_centers.csv`, `cluster_sizes.csv`, `strict_anomalies_clustered/`) plus `07b`'s K=4 self-check under `selfcheck_k4/` |
| `data/output/k_selection_diagnostics/` | `k_selection_diagnostics/run_k_selection_probe.py` (a controlled re-run of `03`'s sweep, kept outside `code/` on purpose) | `k_probe.csv` — the same labels scored under both silhouette scales side by side; the evidence behind the scale note in §8. See its own `README.md` |
| `data/output/graph/` | `ml/analysis/09_eda_clustering.py` (`feature_distributions`, `outlierScore_distribution`, `pca_projection`) and `ml/analysis/08_feature_viz.py` (`viz_boxplots`, `viz_hist_density`, `viz_pairplot`) | Six descriptive figures; read by no script, cite freely |
| `data/output/unit_check/`, `if_repro_probe/`, `out_of_use/` | `ml/tools/`, diagnostics, superseded runs | Provenance only. `out_of_use/` holds superseded outputs — **never cite** |

## 4. How to run it

```bash
# 1. download the 24 monthly files (11.6 GB)
python code/download.py

# 2. clean and profile
python code/preprocessing/preprocessing.py
python code/preprocessing/eda_and_profiling.py

# 3. stratified 10% sample → IsolationForest + SHAP → clustering labels
python code/ml/pipeline/01_sample_v2.py
python code/ml/pipeline/02b_isolation_forest_shap_rowkeyed.py
python code/ml/pipeline/03_kmeans_cluster_anomalies.py     # LEGACY K=2 labels
python code/ml/analysis/10_robust_clustering_rowkeyed.py   # authoritative K sweep + labels
python code/ml/pipeline/04_rename_clusters.py              # numeric ids → business names

# 4. summaries and figures
python code/ml/analysis/05_cluster_money_summary.py
python code/ml/analysis/11_final_shap_plots.py
python code/ml/pipeline/06_plot_cluster_viz.py
python code/ml/analysis/08_feature_viz.py

# 5. optional: regenerate the parallel-scaling figures/checks from persisted CSVs (no Spark)
python code/ml/experiments/parallel_scaling/run_scaling_study.py --analysis-only
```

The full scaling study (2 M/1 M subsets → K1/K2 → IF → plots) is
`python code/ml/experiments/parallel_scaling/run_scaling_study.py`; add
`--sandbox-temp` only in the DSH Windows sandbox if the JVM gateway cannot create
its temporary directory.

Tested with **Python 3.11 / 3.14, PySpark 4.x, single node `local[*]`**. The full pipeline is
CPU-only; no cluster and no GPU are required.

## 5. Parallel strategy in one paragraph

Four things are done in parallel rather than one: **(i) I/O** — Parquet is read in 24 monthly
files and written partitioned by date, so every downstream job scans only what it needs, and the
final BI layer is pre-aggregated to 731 daily partitions (558 MB against 11.6 GB raw);
**(ii) computation** — cleaning, the two sampling stages, the SHAP value computation and
KMeans run as Spark jobs over `local[*]` partitions. The IsolationForest **fit itself is
single-process**: `02b` collects the 4.67 M sampled rows with `toPandas()` and calls sklearn
`IsolationForest(n_estimators=150, max_samples=512, max_features=0.8, n_jobs=4)`, so only the
read, filter and stratified sampling ahead of it are distributed; **(iii) skew** — trips are stratified over six `(demand layer × platform)`
strata, and EDA showed 123 of 262 pickup zones account for 80% of volume, so grouping is done on
the pre-aggregated layer to avoid shuffle skew; **(iv) model fitting** — the IsolationForest
forest is built on the driver with `n_jobs=4` joblib workers (thread-level parallelism over the
150 trees, not executor-level), and K selection loops seeds and K values in-process while each
individual KMeans fit runs inside Spark.

Honest limit: the old sub-5-second micro-benchmarks (partition size, core count, column pruning)
are dominated by fixed Spark scheduling overhead and are **not** usable as speed-up evidence. The
v3 study therefore uses 2 M-row (KMeans/BisectingKMeans) and 1 M-row (IF scoring) subsets with
repeated runs. It finds that **KMeans(k=2) uses `local[4]`** (1.44–1.63× vs `local[2]`;
`local[8]` gives no further wall-clock gain), **BisectingKMeans(k=8) uses `local[8]`**
(primary n=4 1.79–1.89×; a post-hoc S=30 replication gives 1.77× pooled, so quote
“about 1.8×”, not a stable 2.2×),
and **IF `decision_function` peaks at `n_jobs=8`**, not at 20 threads.
The full-sample K=8 fit at **2,198 s** (`data/log/kmeans_full.log`) remains a method-selection
timing, not a production speed-up; the 471 M-row production configuration is still unmeasured.
Details and caveats: `data/output/scaling_study/SUMMARY.md` and
`data/output/scaling_study/README.md`; artefacts in `data/output/scaling_study/`.

## 6. Key modelling decisions

- **Flag, then filter, not `dropna`.** The cleaning stage flags every row and removes nothing; the
  unsupervised track then keeps rows that pass all three flags, which retains 99.0% of the downloaded
  rows;
  median-imputing fare or distance would have manufactured the very anomalies the project is
  looking for.
- **Two-stage sampling (10% then 10%).** 467 M rows × 9 features does not fit in memory at once, so
  a stratified 10% sample is drawn first (`01_sample_v2.py` → `sampled_km/`, 46,706,095 rows) and a
  second 10% is taken from that inside `02b` (4,672,734 rows), keeping the heavier per-row work
  (SHAP, KMeans, bootstrap) at ~4.7 M rows while the full sample still serves the K=8 full-data fit.
  Two bookkeeping details belong in the report rather than being left implicit:
  (a) the 10% is applied **per stratum with a 30-row floor** (`MIN_SAMPLES_PER_LAYER`); every one of
  the six strata is large enough that the floor never binds, and each is sampled at exactly 10.00%
  (cold 3,746,965 / warm 8,066,623 / hot 34,892,507), so the sample keeps the population ratio
  8.0% / 17.3% / 74.7%; (b) between the validation filter (467,142,106 rows) and the sampling input
  (467,067,101 rows) **75,005 rows (0.016%) are dropped**, which is consistent with the
  `1 ≤ speed_mph ≤ 80` filter at `01_sample_v2.py:126-127` but was never printed to the log, so it
  is an inference from the code rather than a logged fact.
- **Explicit threshold instead of `contamination`.** The threshold is set from the empirical IQR of
  the anomaly score, not from a distributional assumption, and is verified on the full sample
  (13,577 flagged = 0.291%), which keeps the flag rate auditable.
- **`row_uid` before any join.** An earlier SHAP join matched rows *by position*, producing cluster
  driver rankings that were effectively random. The pipeline now materialises a `row_uid` in 02b,
  and the SHAP table is joined on it with an explicit assertion that no row is lost.
- **K from stability, not from a single fit.** A single KMeans fit gave a silhouette of 0.3662 at
  K=2 on an earlier table without `row_uid`. That figure is not wrong — it is Spark's
  `squaredEuclidean` silhouette, an independent re-run reproduces it (0.364) — but it sits on a
  different scale from the Euclidean silhouette reported everywhere else (0.2198); the ratio is
  1.66-1.70 across K=2..8, not an exact constant, because a squared distance is not a linear rescaling
  of a metric. The two must never be compared, and the old sweep table is filed under
  `data/output/out_of_use/_outputs/anomaly_clusters_positional_jsonvalid/` so that it cannot be
  quoted by accident. Ten seeds × K=2…8 (`n_init=50`) all select K=2. The two cluster partitions
  from the legacy single fit and the robust sweep agree **row for row** (verified on a nine-column
  feature fingerprint, ARI = 1.000), so the partition itself is not sensitive to initialisation —
  only the metric was.
  For the stability number we refit KMeans on each of 20 bootstrap resamples and score ARI **only
  on the trips that draw left out** (~4,979 of 13,577 per draw): 0.9593 ± 0.0283, minimum 0.9107.
  Scoring on in-bag points instead would only measure how well a model reproduces its own training
  labels, which is why the out-of-bag variant is the one reported.
- **Why K=2 and not K=4.** The mean silhouette is not monotone in K: K=3 is 0.1833 but K=4 is
  0.1870, and a naive "highest silhouette" reading of that pair would look at K=4. Two things rule
  it out: (i) both values sit ~0.033 below K=2 (0.2198), so there is no peak there at all, only a
  local ripple on a gently declining plateau; (ii) K=4's extra split is the one predicted by the
  PCA plot — it separates the **zero-tip subgroup** inside `long_low_tip` (2,130 of its 3,786 trips
  have `tips = 0`), i.e. it carves a degenerate zero-mass group out of one cluster rather than
  discovering a new behavioural family. `anomaly_clusters_rowkeyed/selfcheck_k4/` keeps one early
  run that did select K=4; its numbers come from a version of the script that is no longer in the
  repository and cannot be reproduced (re-running today's code gives K=2 the highest score), so it
  is historical only and its sweep table is not cited. See `selfcheck_k4/README.md`.
- **Why the 13,577 flags are plausible without ground truth.** Three properties hold that a
  random subset of 4.67 M trips would not exhibit:
  *(a) the flags concentrate where trips are extreme.* Median distance of a flagged trip is
  **38.99 mi against 3.00 mi population-wide (13.0×)**, median fare **$150.08 vs $18.93 (7.9×)**,
  and **27.66% of flagged trips exceed 50 mi against 0.1303% of all cleaned trips (212×)**.
  *(b) the flag rate is broadly flat across demand layers.* If the detector were merely reacting to
  busy areas, the hot stratum would dominate out of proportion to its size. Measured against each
  stratum's full size in `sampled_km`, the rates are **0.357% (hot), 0.120% (cold) and 0.084%
  (warm)** — a hot/cold ratio of only 2.95×, and **4.78× across all six `(layer × platform)`
  strata** (0.382% for `hot × HV0003` down to 0.080% for `warm × HV0003`). The hot stratum does
  hold 91.7% of the flags, but it also holds 74.7% of the scored rows, so the flags are not
  concentrating beyond what the volume already implies. (Note: every stratum's rate is below the
  pooled 0.291%, which is not a contradiction — the pooled figure is their weighted mean, so the
  maximum-size stratum necessarily sits on the high side of it.)
  *(c) the threshold is derived, not assumed.* The cut is the empirical IQR of the score
  distribution (−0.1664) rather than a `contamination` prior, and the probe re-derives it from the
  persisted mask (−0.166403) independently.
  What this does **not** establish is precision: the flags are *internally coherent and land on
  extreme trips*, which is necessary but not sufficient for them to be *erroneous* trips.
- **Units verified, not assumed.** `log_duration` is in minutes and `speed_mph` in physical mph,
  established with the algebraic identity `log1p((miles / speed_mph) × 60) == log_duration`
  (max residual 8.9×10⁻¹⁶ over 2 M rows; the hour hypothesis is off by 3.49 log-units).

## 7. Known limitations (ML scope)

1. Parallel **speed-up** is measured only on **2 M/1 M subsets**, not on the 471 M-row production
   input; the old sub-5-second benchmarks remain noise-dominated (see §5). The 2 M subset says
   K1=4, K2=8 (1.8–1.9×; S=30 CI lower 1.23), IF=8; a post-hoc S=30 K2 replication did not
   reproduce the 143.8 s point, but its median was 10.1% slower and pooled 1.77×, so the S=30
   estimate remains uncertain. A 10 M-row re-test is the next step before changing production settings.
2. No supervised model yet. The feature table and cluster labels needed for an XGBoost baseline
   are in place (`data/output/anomaly_features_v2/anomalies_enriched_v2.parquet`), but the model
   has not been built. A calibration pass (`data/output/xgb_cost_calibration/`) showed that neither
   data preparation nor training cost is the obstacle on this machine — a full 471.7 M-row feature
   scan takes 3.8 s and a single-node boosting round 0.7–1.2 s — so what is missing is time to close
   the loop (features, baselines, model, evaluation, parallel evidence), not capability.
3. IsolationForest has **no baseline comparison** (e.g. against a rule-only or LOF baseline), so the
   0.291% flag rate is internally consistent but not externally validated.
4. **Single node only** (the counterpart of 1). Everything runs inside one JVM on `local[*]`, so no
   shuffle crosses a network: scaling *out* is untested and the Amdahl discussion is confined to
   cores inside one process.
5. **No labelled ground truth, so precision and recall are not reported.** The anomaly set is
   supported by internal evidence only — the `row_uid` rerun reproduces the same 13,577 flags, and
   §6 shows they land on extreme trips and concentrate in the hot stratum. A hand-checked sample
   (say 200 flagged and 200 unflagged trips) would supply the missing external validation; that was
   not done in this scope and is listed as next step 4.

## 8. Provenance

Every number in the report is traceable to a file under `data/output/` or `data/log/`.
Logs of superseded runs are kept under `data/log/out_of_use/` together with a per-log provenance
table (`data/log/out_of_use/README.md`), and superseded artefacts under
`data/output/out_of_use/` — neither should be cited.

One join is **positional, not keyed**, and that must be stated wherever it is used:
`data/output/anomaly_clusters_rowkeyed/strict_anomalies_clustered/` carries no `row_uid` column.
It is reproducible because `03` reads `if_results_rowkeyed/strict_anomalies_rowkeyed` in one Spark
job, so the two tables share the same row order — a `row_uid` assigned by position then matches
13,577/13,577 rows against `clustering_robustness/row_uid_cluster_robust.parquet` (ARI = 1.000).
Any upstream change to `02b` (different sampling seed, filter, or partition order) breaks that
alignment silently, so the positional assignment must be re-verified, not assumed.

That positional assumption has since been **checked rather than relied on**. The nine raw feature
columns are unique across all 13,577 rows (no duplicates; `outlierScore` alone is *not* unique, with
574 tied values, so it cannot serve as a key). Joining the unkeyed cluster table to the upstream
table on that nine-column fingerprint matches **13,577/13,577 rows with nothing on either side
left over**, and the resulting `row_uid` links to the authoritative table with no loss. The
cross-tabulation is fully diagonal: `03`'s cluster 0 is `short_high_tip` (9,791 trips) and its
cluster 1 is `long_low_tip` (3,786). So the legacy single-fit partition and the 10-seed robust
partition are **the same partition row for row**, and the cluster numbering of the two is reversed
(`03`: 0 = `short_high_tip`; `10`: 0 = `long_low_tip`) — always quote cluster names, never numbers.

Two silhouette scales exist in this repository and must never be compared. Spark's
`ClusteringEvaluator` accepts only `distanceMeasure="squaredEuclidean"` (passing `"euclidean"` raises
`IllegalArgumentException`), so `03`'s K sweep reports a squared-distance silhouette: 0.364 at K=2 on
the keyed table, reproduced by an independent re-run. `clustering_robustness/k_sweep.csv` reports
sklearn's Euclidean silhouette: 0.220 at K=2. The squared-Euclidean score exceeds the Euclidean one
by a factor between 1.66 and 1.70 for every K from 2 to 8 - approximately, not exactly, constant
(a squared distance is not a linear rescaling of a metric), so the older 0.3662 figure recorded from
the *unkeyed* table is a scale difference, not a corrupted value — but it is on a different scale from every other number in this README and is filed
under `data/output/out_of_use/_outputs/anomaly_clusters_positional_jsonvalid/` for that reason. Cite
`clustering_robustness/` for model selection; never place the two side by side. The two scales side
by side on one set of labels, together with the feature-norm audit that rules out degenerate rows as
an explanation, are in `data/output/k_selection_diagnostics/`.

One artefact is **untraced**: an early KMeans tuning sweep whose parameters cannot be reconstructed
from the repository and whose magnitudes match no logged run. It has been moved out of the live
tree to `data/output/out_of_use/_outputs/tuning_results_untraced/` and is excluded from every
result in this README. See `data/output/out_of_use/README.md` for the full audit.

## 9. Next steps

1. **Parallel scaling study** — **done for 2 M/1 M subsets** (`data/output/scaling_study/`,
   code in `code/ml/experiments/parallel_scaling/`); K1 uses `local[4]`, K2 `local[8]`
   (1.8–1.9× vs 2 cores), IF `n_jobs=8`. Remaining: re-test at 10 M rows before changing the production `local[*]` /
   `n_jobs=-1` settings. See `data/output/scaling_study/README.md`.
2. **Supervised baseline** — an XGBoost classifier on `anomalies_enriched_v2.parquet` to test
   whether the two behavioural groups are separable by a supervised model; the plan is
   summarised in the report's future-work section. Cost calibration is already done
   (`data/output/xgb_cost_calibration/`, ~6–8 h of engineering left), so this is deferred work, not
   unproven infrastructure.
3. **Anomaly baseline** — a rule-only or LOF baseline for the IsolationForest flag rate.
4. **External validation of the flags** — hand-label ~200 flagged and ~200 unflagged trips to
   estimate precision; this is the one missing piece that would turn "internally consistent" into
   "externally validated".
