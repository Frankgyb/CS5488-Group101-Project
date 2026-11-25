"""
Location: project/code/ml/analysis/07b_shap_cluster_analysis_rowkeyed.py | PROJECT_ROOT depth: 3
07b_shap_cluster_analysis_rowkeyed.py - Per-cluster SHAP analysis, KEY-BASED.

REPLACES `07_shap_cluster_analysis.py`, whose alignment logic was unsound:

    07:48-51   df_clustered = df_clustered.iloc[:2000].reset_index(drop=True)
    07:52      df_shap      = df_shap.reset_index(drop=True)
    07:55      df_shap['cluster'] = df_clustered['cluster'].values

`strict_anomalies_clustered` is written by 03 as a `groupBy("cluster")` result
(03:104), i.e. sorted by cluster, while `shap_strict_anomalies.csv` was written
by 02 in the original sampling order (`anom_pdf.head(2000)`, mixed clusters).
The two "first 2000 rows" therefore describe DIFFERENT trips, so the labels
were effectively random. Every conclusion derived from that join is void.

This version joins on `row_uid`, which 02b carries from the raw read through
SHAP, and then runs 03's clustering on exactly those rows. Row-for-row
alignment is asserted, not assumed.

INPUTS
  - data/output/if_results_rowkeyed/strict_anomalies_rowkeyed   (row_uid + features)
  - data/output/if_results_rowkeyed/shap_rowkeyed.csv           (row_uid + shap_*)

OUTPUTS (data/output/anomaly_clusters_rowkeyed/selfcheck_k4/)
  - strict_anomalies_clustered_rowkeyed/   parquet: row_uid + features + cluster + cluster_name
  - shap_cluster_summary.csv               mean |SHAP| and mean signed SHAP per cluster
  - shap_top3_per_cluster.csv              top-3 drivers per cluster
  - shap_bar_cluster_{k}.png               signed-SHAP bar plot per cluster
  - shap_beeswarm_cluster_{k}.png          per-cluster beeswarm (direction + magnitude)
  - alignment_audit.csv                    explicit audit of the join
  - k_selection_summary.csv                silhouette per K (recomputed)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # .../project

IF_DIR = PROJECT_ROOT / "data" / "output" / "if_results_rowkeyed"
ANOM_PATH = IF_DIR / "strict_anomalies_rowkeyed"
SHAP_PATH = IF_DIR / "shap_rowkeyed.csv"
META_PATH = IF_DIR / "run_meta.json"

# SELF-CHECK OUTPUT (moved 2026-11-22, was out_of_use/_outputs/
# kmeans_anomaly_clusters_rowkeyed_deprecated/): this run includes 07b's OWN
# in-script K sweep, which picks K=4 on the rowkeyed anomalies. Its cluster
# labels are therefore NOT the reported model and must never be quoted as such;
# they sit in a `selfcheck_k4` sub-folder next to the live K=2 output so the two
# are impossible to confuse. Authoritative labels:
# data/output/clustering_robustness/row_uid_cluster_robust.parquet
# (10 seeds x n_init=50 -> K=2, silhouette 0.2198, bootstrap ARI 0.96).
# Consumers only take the row_uid-keyed parquet from here (10_robust and
# add_consistency_features_v2 run their own clustering / merge the robust labels).
OUT_DIR = (PROJECT_ROOT / "data" / "output" / "anomaly_clusters_rowkeyed" /
           "selfcheck_k4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "log_miles", "log_fare", "log_tips", "speed_mph", "log_duration",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
]
SHAP_COLS = [f"shap_{c}" for c in FEATURE_COLS]

# Business names, from 04_rename_clusters.py:23-26 (the single source of truth).
CLUSTER_MAP = {0: "short_high_tip", 1: "long_low_tip"}
CLUSTER_DESC = {
    0: "short trip, low fare, high tip",
    1: "long trip, high fare, near-zero tip",
}

audit = []


def log(msg=""):
    print(msg)
    audit.append(str(msg))


# ------------------------------------------------------------
# 1. Load both sides
# ------------------------------------------------------------
log("=" * 78)
log("KEY-BASED SHAP x CLUSTER ALIGNMENT")
log("=" * 78)

anom = pd.read_parquet(ANOM_PATH)
shap_df = pd.read_csv(SHAP_PATH)

log(f"strict_anomalies_rowkeyed : {anom.shape[0]:,} rows x {anom.shape[1]} cols")
log(f"shap_rowkeyed.csv         : {shap_df.shape[0]:,} rows x {shap_df.shape[1]} cols")

if META_PATH.exists():
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    log(f"run_meta: anomalies={meta['anomalies']:,} "
        f"(matches 02: {meta['anomaly_count_matches_02']}), "
        f"shap_samples={meta['shap_samples']:,}")

# ------------------------------------------------------------
# 2. Integrity: row_uid must be unique on both sides and fully present in anom
# ------------------------------------------------------------
log("")
log("-- integrity checks --")
uid_anom_unique = anom["row_uid"].is_unique
uid_shap_unique = shap_df["row_uid"].is_unique
log(f"row_uid unique in anomalies : {uid_anom_unique}")
log(f"row_uid unique in SHAP      : {uid_shap_unique}")
assert uid_anom_unique, "duplicate row_uid in strict_anomalies_rowkeyed"
assert uid_shap_unique, "duplicate row_uid in shap_rowkeyed.csv"

missing = ~shap_df["row_uid"].isin(set(anom["row_uid"]))
log(f"SHAP row_uids missing from anomalies : {int(missing.sum())}")
assert missing.sum() == 0, "SHAP contains row_uid not present in anomalies"

# cross-check the features travelled together with the uid
# In shap_rowkeyed.csv the raw feature value is stored as `feat_<name>`,
# in the anomaly table it is simply `<name>`.
left = shap_df[["row_uid"] + [f"feat_{c}" for c in FEATURE_COLS]].rename(
    columns={f"feat_{c}": f"{c}_shap" for c in FEATURE_COLS})
right = anom[["row_uid"] + FEATURE_COLS].rename(
    columns={c: f"{c}_anom" for c in FEATURE_COLS})
feat_check = left.merge(right, on="row_uid", how="inner")
log(f"rows compared for feature equality : {len(feat_check):,}")
max_feat_diff = {}
for c in FEATURE_COLS:
    d = np.nanmax(np.abs(feat_check[f"{c}_shap"] - feat_check[f"{c}_anom"]))
    max_feat_diff[c] = float(d)
worst = max(max_feat_diff, key=max_feat_diff.get)
log(f"max feature mismatch after join : {max_feat_diff[worst]:.3e} ({worst})")
assert max_feat_diff[worst] < 1e-9, (
    f"row_uid does not identify the same trip: {worst} differs by "
    f"{max_feat_diff[worst]:.3e}")
log("=> row_uid join verified: SHAP rows and anomaly rows are the SAME trips.")

# ------------------------------------------------------------
# 3. Recommend a K, using the same procedure as 03
# ------------------------------------------------------------
log("")
log("-- KMeans K selection (mirrors 03 pipeline) --")
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

spark = (SparkSession.builder
         .appName("07b_shap_cluster_rowkeyed")
         .master("local[8]")
         .config("spark.driver.memory", "8g")
         .config("spark.sql.shuffle.partitions", "16")
         .config("spark.local.dir", "D:/temp_spark")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

sdf = spark.createDataFrame(anom)
assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw")
sdf = assembler.transform(sdf)
scaler = SparkScaler(inputCol="features_raw", outputCol="features",
                     withStd=True, withMean=True)
sdf = scaler.fit(sdf).transform(sdf)

evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster",
                                metricName="silhouette",
                                distanceMeasure="squaredEuclidean")
k_rows = []
best_k, best_sil, best_model = None, -1.0, None
for k in range(2, 9):
    km = KMeans(featuresCol="features", predictionCol="cluster",
                k=k, seed=42, maxIter=20, tol=1e-4)
    m = km.fit(sdf)
    sil = evaluator.evaluate(m.transform(sdf))
    k_rows.append({"k": k, "wssse": round(float(m.summary.trainingCost), 2),
                   "silhouette": round(float(sil), 4)})
    log(f"   K={k}: WSSSE={k_rows[-1]['wssse']:.2f}  silhouette={sil:.4f}")
    if sil > best_sil:
        best_sil, best_k, best_model = sil, k, m

log(f"best K = {best_k} (silhouette {best_sil:.4f})")
pd.DataFrame(k_rows).to_csv(OUT_DIR / "k_selection_summary.csv", index=False)

lab = best_model.transform(sdf).select("row_uid", "cluster").toPandas()

# ------------------------------------------------------------
# 4. The join that 07 got wrong
# ------------------------------------------------------------
log("")
log("-- alignment --")
n_before = len(shap_df)
merged = shap_df.merge(lab, on="row_uid", how="inner")
log(f"SHAP rows           : {n_before:,}")
log(f"rows after join     : {len(merged):,}")
assert len(merged) == n_before, (
    f"join dropped {n_before - len(merged)} SHAP rows - alignment incomplete")
log("join is 1:1 with no loss => every SHAP value now has its OWN cluster label.")

merged["cluster_name"] = merged["cluster"].map(CLUSTER_MAP)
dist = merged["cluster"].value_counts().sort_index()
log("cluster distribution inside the SHAP sample:")
for cid, cnt in dist.items():
    log(f"   cluster {cid} ({CLUSTER_MAP.get(cid, '?')}): {cnt:,} "
        f"({cnt/len(merged)*100:.1f}%)")

# ------------------------------------------------------------
# 5. Mean SHAP per cluster
# ------------------------------------------------------------
log("")
log("-- mean |SHAP| per cluster --")
rows = []
for cid in sorted(merged["cluster"].unique()):
    sub = merged[merged["cluster"] == cid]
    for c in FEATURE_COLS:
        col = f"shap_{c}"
        rows.append({
            "cluster": int(cid),
            "cluster_name": CLUSTER_MAP.get(int(cid)),
            "feature": c,
            "n": len(sub),
            "mean_abs_shap": round(float(sub[col].abs().mean()), 6),
            "mean_signed_shap": round(float(sub[col].mean()), 6),
        })
summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "shap_cluster_summary.csv", index=False)

top3 = (summary.sort_values(["cluster", "mean_abs_shap"], ascending=[True, False])
        .groupby("cluster").head(3).reset_index(drop=True))
top3.to_csv(OUT_DIR / "shap_top3_per_cluster.csv", index=False)

log(f"{'cluster':>7} {'name':>15} {'top-3 drivers (mean |SHAP|)':<60}")
for cid in sorted(top3["cluster"].unique()):
    sub = top3[top3["cluster"] == cid]
    bits = ", ".join(f"{r.feature}={r.mean_abs_shap:.3f}"
                     for r in sub.itertuples())
    log(f"{cid:>7} {CLUSTER_MAP.get(cid,''):>15} {bits:<60}")

# ------------------------------------------------------------
# 6. Plots
# ------------------------------------------------------------
log("")
log("-- plots --")
for cid in sorted(merged["cluster"].unique()):
    sub = merged[merged["cluster"] == cid]

    # signed bar plot
    mean_signed = sub[SHAP_COLS].mean()
    order = mean_signed.sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#c0392b" if v < 0 else "#2471a3" for v in order.values]
    ax.barh([c.replace("shap_", "") for c in order.index], order.values,
            color=colors)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Mean signed SHAP value (log-odds of anomaly score)")
    ax.set_title(f"Cluster {cid} - {CLUSTER_MAP.get(cid,'')}\n"
                 f"n={len(sub):,} anomalies | {CLUSTER_DESC.get(cid,'')}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"shap_bar_cluster_{cid}.png", dpi=150)
    plt.close(fig)

    # beeswarm: direction + magnitude + feature value
    try:
        import shap as shaplib
        fig = plt.figure(figsize=(9, 5.5))
        shaplib.summary_plot(
            sub[SHAP_COLS].to_numpy(),
            sub[[f"feat_{c}" for c in FEATURE_COLS]].to_numpy(),
            feature_names=FEATURE_COLS,
            show=False, plot_size=None,
        )
        plt.title(f"Cluster {cid} - {CLUSTER_MAP.get(cid,'')} (n={len(sub):,})")
        plt.tight_layout()
        fig.savefig(OUT_DIR / f"shap_beeswarm_cluster_{cid}.png", dpi=150)
        plt.close(fig)
    except Exception as e:                       # pragma: no cover
        log(f"   beeswarm for cluster {cid} skipped: {e}")

    log(f"   saved shap_bar_cluster_{cid}.png / shap_beeswarm_cluster_{cid}.png")

# ------------------------------------------------------------
# 7. Persist clustered anomalies with BOTH ids and names
# ------------------------------------------------------------
out_df = anom.merge(lab, on="row_uid", how="left")
out_df["cluster_name"] = out_df["cluster"].map(CLUSTER_MAP)
out_path = OUT_DIR / "strict_anomalies_clustered_rowkeyed"
spark.createDataFrame(out_df).write.mode("overwrite").parquet(str(out_path))
log("")
log(f"saved clustered anomalies -> {out_path}")
n_unlabelled = int(out_df["cluster"].isna().sum())
log(f"unlabelled rows after clustering: {n_unlabelled}")
assert n_unlabelled == 0

# ------------------------------------------------------------
# 8. Audit + comparison against the old (invalid) numbers
# ------------------------------------------------------------
log("")
log("-- comparison with the OLD 07 output (invalid, position-based) --")
# 2026-11-22: 03's positional-join K=2 table (VOID numbers, kept only as the
# "old" side of the correction figure) was moved out of the live tree so that
# nobody reads it in place. The live anomaly_clusters_rowkeyed/ now holds only
# 03's row-keyed cluster tables.
old_top3 = (PROJECT_ROOT / "data" / "output" / "out_of_use" / "_outputs" /
            "anomaly_clusters_positional_jsonvalid" / "shap_top3_per_cluster.csv")
if old_top3.exists():
    old = pd.read_csv(old_top3)
    old_groups = old[old["cluster"] == 0]["cluster"]
    log("OLD (void):")
    for cid in sorted(old["cluster"].unique()):
        feats = ", ".join(old[old['cluster'] == cid]['feature'].tolist())
        log(f"   cluster {cid}: {feats}")
    log("NEW (key-joined):")
    for cid in sorted(top3["cluster"].unique()):
        feats = ", ".join(top3[top3['cluster'] == cid]['feature'].tolist())
        log(f"   cluster {cid}: {feats}")
    same = all(
        old[old["cluster"] == c].head(3)["feature"].tolist()
        == top3[top3["cluster"] == c].head(3)["feature"].tolist()
        for c in top3["cluster"].unique()
        if c in set(old["cluster"])
    )
    log(f"identical conclusions: {same}")
else:
    log("   old output not found - nothing to compare")

audit_df = pd.DataFrame({"line": audit})
audit_df.to_csv(OUT_DIR / "alignment_audit.csv", index=False)
(OUT_DIR / "alignment_audit.txt").write_text("\n".join(audit), encoding="utf-8")

spark.stop()
log("")
log(f"All outputs in {OUT_DIR}")
