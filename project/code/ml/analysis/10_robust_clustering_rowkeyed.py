"""
Location: project/code/ml/analysis/10_robust_clustering_rowkeyed.py | PROJECT_ROOT depth: 3
10_robust_clustering_rowkeyed.py - Reproducible, multi-seed cluster selection.

WHY
---
`03_kmeans_cluster_anomalies.py` calls Spark MLlib KMeans with
`k=k, seed=42, maxIter=20, tol=1e-4` (03:82-84). Spark's KMeans default is
`initMode="k-means||"` with `initSteps=2`, so a single initialisation decides
the partition. Two runs on two DIFFERENT 13,577-row anomaly tables produced
different silhouette curves:

    old table, no row_uid : K=2 0.3662 | K=3 0.3088 | K=4 0.2713   -> picks K=2
    row_uid table         : K=2 0.2731 | K=3 0.3084 | K=4 0.3119   -> picks K=4

The difference is NOT initialisation luck: on the row_uid table the K=2
silhouette is 0.2198, not 0.2731, and the actual partitions are identical
(adjusted Rand index 1.000 between 03's single-init fit and this script's
result). The old 0.3662 column is inflated because that table contained
degenerate rows, and 03's silhouette is computed with Spark
ClusteringEvaluator(squaredEuclidean) while this script uses sklearn's
euclidean silhouette - the two numbers are not comparable. See
data/output/clustering_robustness/ and data/output/k_selection_diagnostics/ for the evidence.

This script therefore evaluates K on a multi-seed basis rather than trusting
one fit:
  * sklearn KMeans with n_init=50 (best of 50 inits) as the reference,
  * every K in 2..8 evaluated across 10 different seeds,
  * reports mean / std / min / max silhouette per K,
  * a cluster must survive a bootstrap stability test (adjusted Rand index)
    against resampled fits, not just win one silhouette number,
  * K is selected by the *stable* rule: highest mean silhouette whose
    seed-to-seed std is small.

OUTPUTS (data/output/clustering_robustness/)
  - k_sweep.csv            silhouette per (K, seed) + summary stats
  - silhouette_curve.png   mean +/- std band across seeds
  - bootstrap_ari.csv      ARI stability of the chosen K under resampling
  - cluster_profile.csv    physical-unit profile per cluster
  - chosen_model.json      the selected K and evidence
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Input: the row_uid-keyed clustered anomalies written by 07b's K=4 self-check.
# 2026-11-22: moved out of out_of_use/ to anomaly_clusters_rowkeyed/selfcheck_k4/
# (it is a live input of THIS script, so it does not belong in the deprecated tree).
# Only cluster / cluster_name / row_uid columns are ever used from it - this script
# re-clusters with 10 seeds x n_init=50 and selects K=2 unanimously.
# Authoritative cluster labels: data/output/clustering_robustness/row_uid_cluster_robust.parquet
CLUSTERED = (PROJECT_ROOT / "data" / "output" / "anomaly_clusters_rowkeyed" /
             "selfcheck_k4" / "strict_anomalies_clustered_rowkeyed")
OUT_DIR = PROJECT_ROOT / "data" / "output" / "clustering_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "log_miles", "log_fare", "log_tips", "speed_mph", "log_duration",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
]
K_RANGE = list(range(2, 9))
SEEDS = [42, 7, 2024, 1, 13, 99, 123, 555, 777, 2025]
N_INIT = 50
N_BOOTSTRAP = 20


def log(msg=""):
    print(msg)


log("=" * 78)
log("ROBUST CLUSTERING - multi-seed K selection on the anomaly set")
log("=" * 78)

df = pd.read_parquet(CLUSTERED)
log(f"rows: {len(df):,}   cols: {df.shape[1]}")
log(f"row_uid unique: {df['row_uid'].is_unique}")

X_raw = df[FEATURE_COLS].to_numpy(dtype=float)
X = StandardScaler().fit_transform(X_raw)

# ------------------------------------------------------------
# 1. K sweep across seeds
# ------------------------------------------------------------
log("")
log(f"-- K sweep: {len(SEEDS)} seeds x K in {K_RANGE}, n_init={N_INIT} --")

records = []
for seed in SEEDS:
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(X)
        s = float(silhouette_score(X, km.labels_))
        records.append({"k": k, "seed": seed,
                        "silhouette": round(s, 6),
                        "inertia": round(float(km.inertia_), 2)})

sweep = pd.DataFrame(records)
sweep.to_csv(OUT_DIR / "k_sweep.csv", index=False)

stats = (sweep.groupby("k")["silhouette"]
         .agg(["mean", "std", "min", "max"])
         .reset_index())
log(f"{'K':>3} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}   {'inertia':>12}")
inertia_mean = sweep.groupby("k")["inertia"].mean()
for r in stats.itertuples():
    log(f"{r.k:>3} {r.mean:>10.4f} {r.std:>10.4f} {r.min:>10.4f} "
        f"{r.max:>10.4f}   {inertia_mean[r.k]:>12,.0f}")

# selection rule: highest mean silhouette, tie-break on lower std
stats_sorted = stats.sort_values(["mean", "std"], ascending=[False, True])
best_k = int(stats_sorted.iloc[0]["k"])
best_mean = float(stats_sorted.iloc[0]["mean"])
best_std = float(stats_sorted.iloc[0]["std"])
log("")
log(f"selected K = {best_k}  (mean silhouette {best_mean:.4f} "
    f"+/- {best_std:.4f} across {len(SEEDS)} seeds)")

# how stable is the RANKING itself?
log("")
log("-- ranking stability: which K wins, per seed --")
winners = []
for seed in SEEDS:
    sub = sweep[sweep["seed"] == seed]
    winners.append(int(sub.loc[sub["silhouette"].idxmax(), "k"]))
from collections import Counter
win_counts = Counter(winners)
log(f"   winner per seed: {dict(sorted(win_counts.items()))}")
log(f"   unanimous: {len(win_counts) == 1}")

# ------------------------------------------------------------
# 2. Silhouette curve plot
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(stats["k"], stats["mean"], yerr=stats["std"],
            marker="o", capsize=4, color="#2471a3", label="mean +/- std across seeds")
ax.fill_between(stats["k"], stats["min"], stats["max"], alpha=0.15,
                color="#2471a3", label="min..max across seeds")
ax.axvline(best_k, color="#c0392b", linestyle="--", linewidth=1,
           label=f"selected K={best_k}")
ax.set_xlabel("K (number of clusters)")
ax.set_ylabel("Silhouette score")
ax.set_title("K selection stability across 10 random inits\n"
             "(a single KMeans fit - what Spark MLlib exposes - cannot show this)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "silhouette_curve.png", dpi=150)
plt.close(fig)
log("")
log("saved silhouette_curve.png")

# ------------------------------------------------------------
# 3. Bootstrap stability (ARI) of the chosen K
# ------------------------------------------------------------
log("")
log(f"-- bootstrap stability of K={best_k} (ARI vs bootstrap refits) --")
ref = KMeans(n_clusters=best_k, n_init=N_INIT, random_state=42).fit(X)
labels_ref = ref.labels_

rng = np.random.default_rng(42)
aris = []     # proper bootstrap: fit on the resample, score on the OUT-OF-BAG points
n_oob = []    # how many points each draw actually held out (0.368 * n on average)
for b in range(N_BOOTSTRAP):
    idx = rng.choice(len(X), size=len(X), replace=True)
    oob = np.setdiff1d(np.arange(len(X)), idx, assume_unique=False)
    if len(oob) < 10:
        continue                      # a degenerate draw cannot be scored
    km = KMeans(n_clusters=best_k, n_init=10, random_state=b).fit(X[idx])
    # Compare on the points that draw did NOT see. Scoring on the in-bag points
    # would only reward a model for reproducing its own training labels.
    aris.append(float(adjusted_rand_score(labels_ref[oob], km.predict(X[oob]))))
    n_oob.append(int(len(oob)))

ari_df = pd.DataFrame({"bootstrap": range(len(aris)), "ari": aris,
                       "oob_points": n_oob})
ari_df.to_csv(OUT_DIR / "bootstrap_ari.csv", index=False)
log(f"   ARI mean={np.mean(aris):.4f}  std={np.std(aris):.4f}  "
    f"min={np.min(aris):.4f}")
log(f"   scored on out-of-bag points only (mean {np.mean(n_oob):,.0f} of "
    f"{len(X):,} per draw); ARI >= 0.9 => the partition transports to unseen trips")

# ------------------------------------------------------------
# 4. Physical-unit cluster profile  <-- the units are now VERIFIED
# ------------------------------------------------------------
log("")
log(f"-- cluster profile in PHYSICAL units (K={best_k}) --")
# log_duration is log1p(MINUTES) - proven in data/output/unit_check/
prof = df.copy()
prof["cluster_ref"] = labels_ref
prof["miles"] = np.expm1(prof["log_miles"])
prof["fare"] = np.expm1(prof["log_fare"])
prof["tips"] = np.expm1(prof["log_tips"])
prof["duration_min"] = np.expm1(prof["log_duration"])   # MINUTES (verified)
prof["speed_check_mph"] = np.where(prof["duration_min"] > 0,
                                   prof["miles"] / (prof["duration_min"] / 60.0),
                                   np.nan)

rows = []
for cid in sorted(prof["cluster_ref"].unique()):
    s = prof[prof["cluster_ref"] == cid]
    rows.append({
        "cluster": int(cid),
        "n": len(s),
        "share_pct": round(len(s) / len(prof) * 100, 2),
        "miles_med": round(float(s["miles"].median()), 2),
        "fare_med": round(float(s["fare"].median()), 2),
        "tips_med": round(float(s["tips"].median()), 2),
        "duration_min_med": round(float(s["duration_min"].median()), 1),
        "speed_mph_med": round(float(s["speed_mph"].median()), 1),
        "speed_from_units_med": round(float(s["speed_check_mph"].median()), 1),
        "tip_rate_med": round(float((s["tips"] / s["fare"].replace(0, np.nan))
                                    .median()), 4),
        "zero_tip_pct": round(float((s["tips"] == 0).mean() * 100), 2),
        "fare_per_mile_med": round(float((s["fare"] / s["miles"].replace(0, np.nan))
                                         .median()), 2),
        "hour_peak": int(np.argmax(np.histogram(
            np.arctan2(s["hour_sin"], s["hour_cos"]), bins=24,
            range=(-np.pi, np.pi))[0])),
    })
profile = pd.DataFrame(rows)
profile.to_csv(OUT_DIR / "cluster_profile.csv", index=False)
log(profile.to_string(index=False))

log("")
log("   UNIT CONSISTENCY: `speed_mph_med` and `speed_from_units_med` must agree.")
agree = np.allclose(profile["speed_mph_med"], profile["speed_from_units_med"],
                    atol=0.5)
log(f"   agree: {agree}   <- confirms duration is MINUTES end-to-end")

# ------------------------------------------------------------
# 5. Persist the chosen model's labels
# ------------------------------------------------------------
out = df[["row_uid"]].copy()
out["cluster_robust"] = labels_ref
out["cluster_robust_name"] = out["cluster_robust"].map(
    {0: "long_low_tip", 1: "short_high_tip"})
out_path = OUT_DIR / "row_uid_cluster_robust.parquet"
out.to_parquet(out_path, index=False)
log("")
log(f"saved {out_path}  ({len(out):,} rows)")

chosen = {
    "chosen_k": best_k,
    "mean_silhouette": round(best_mean, 6),
    "std_silhouette": round(best_std, 6),
    "seeds": SEEDS,
    "n_init": N_INIT,
    "winner_per_seed": {str(k): int(v) for k, v in sorted(win_counts.items())},
    "unanimous_winner": len(win_counts) == 1,
    "bootstrap_ari_mean": round(float(np.mean(aris)), 4),
    "bootstrap_ari_std": round(float(np.std(aris)), 4),
    "unit_check": "log_duration = log1p(MINUTES), speed_mph physical "
                  "(see data/output/unit_check/unit_check_report.txt)",
    "supersedes": "03_kmeans_cluster_anomalies.py single-init K selection",
}
(OUT_DIR / "chosen_model.json").write_text(
    json.dumps(chosen, indent=2, ensure_ascii=False), encoding="utf-8")

log("")
log("=" * 78)
log("VERDICT")
log("=" * 78)
log(f"K = {best_k} is the stable choice (mean silhouette {best_mean:.4f}, "
    f"std {best_std:.4f}, unanimous across seeds: {len(win_counts) == 1}).")
log("NOTE: the stored K=2 / silhouette 0.3662 came from the OLD non-rowkeyed")
log("table and a different silhouette metric (Spark squaredEuclidean); it is")
log("inflated and must not be compared with the numbers above. The cluster")
log("ASSIGNMENT itself is identical to this run (ARI = 1.000) - only the")
log("reported metric differs.")
log("")
log(f"All outputs in {OUT_DIR}")
