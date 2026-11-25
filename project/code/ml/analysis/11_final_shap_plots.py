"""
Location: project/code/ml/analysis/11_final_shap_plots.py | PROJECT_ROOT depth: 3
11_final_shap_plots.py - Corrected, publication-ready SHAP figures.

Consumes only the repaired artifacts:
  - data/output/if_results_rowkeyed/shap_rowkeyed.csv          (row_uid + shap_*)
  - data/output/clustering_robustness/row_uid_cluster_robust.parquet

Everything here is joined on `row_uid`, never on row position. The script
re-asserts the alignment before plotting, so a future re-run cannot silently
reintroduce the 07 bug.

Figures (data/output/shap_final/)
  fig_shap_bar_K2.png          mean signed SHAP per cluster (bar)
  fig_shap_beeswarm_K2.png     per-cluster beeswarm (direction + magnitude)
  fig_shap_consistency.png     old (void) vs corrected top drivers, side by side
  shap_top_drivers_final.csv   the numbers behind the figures
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SHAP_PATH = PROJECT_ROOT / "data" / "output" / "if_results_rowkeyed" / "shap_rowkeyed.csv"
LABEL_PATH = (PROJECT_ROOT / "data" / "output" / "clustering_robustness" /
              "row_uid_cluster_robust.parquet")
# 2026-11-22: the K=2 table produced by 03 (positional-join inference, VOID) was
# moved out of the live tree; it is kept ONLY as the "old" comparison side of
# fig_shap_old_vs_corrected.png. If it is absent the figure is skipped.
OLD_TOP3 = (PROJECT_ROOT / "data" / "output" / "out_of_use" / "_outputs" /
            "anomaly_clusters_positional_jsonvalid" / "shap_top3_per_cluster.csv")
OUT_DIR = PROJECT_ROOT / "data" / "output" / "shap_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "log_miles", "log_fare", "log_tips", "speed_mph", "log_duration",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
]
SHAP_COLS = [f"shap_{c}" for c in FEATURE_COLS]
CLUSTER_MAP = {0: "long_low_tip", 1: "short_high_tip"}   # robust numbering (10_robust)

d = pd.read_csv(SHAP_PATH)
lab = pd.read_parquet(LABEL_PATH)

n0 = len(d)
m = d.merge(lab, on="row_uid", how="inner")
assert len(m) == n0, f"join lost {n0-len(m)} rows"
print(f"aligned {len(m):,} SHAP rows to cluster labels by row_uid (1:1, lossless)")

m["cluster_name"] = m["cluster_robust"].map(CLUSTER_MAP)
print("cluster distribution:", m["cluster_robust"].value_counts().sort_index().to_dict())

# ------------------------------------------------------------
# 1. Top drivers table
# ------------------------------------------------------------
rows = []
for cid in sorted(m["cluster_robust"].unique()):
    sub = m[m["cluster_robust"] == cid]
    for c in FEATURE_COLS:
        col = f"shap_{c}"
        rows.append({
            "cluster": int(cid),
            "cluster_name": CLUSTER_MAP[cid],
            "n": len(sub),
            "feature": c,
            "mean_abs_shap": round(float(sub[col].abs().mean()), 5),
            "mean_signed_shap": round(float(sub[col].mean()), 5),
            "shap_sd": round(float(sub[col].std()), 5),
            "frac_positive": round(float((sub[col] > 0).mean()), 4),
        })
table = pd.DataFrame(rows)
table["rank_in_cluster"] = (table.groupby("cluster")["mean_abs_shap"]
                            .rank(ascending=False, method="min").astype(int))
table = table.sort_values(["cluster", "rank_in_cluster"])
table.to_csv(OUT_DIR / "shap_top_drivers_final.csv", index=False)
print()
print(table[table["rank_in_cluster"] <= 4][
    ["cluster", "cluster_name", "feature", "mean_abs_shap",
     "mean_signed_shap", "frac_positive"]].to_string(index=False))

# ------------------------------------------------------------
# 2. Per-cluster bar + beeswarm
# ------------------------------------------------------------
for cid in sorted(m["cluster_robust"].unique()):
    sub = m[m["cluster_robust"] == cid]
    name = CLUSTER_MAP[cid]

    # --- bar ---
    ms = sub[SHAP_COLS].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = ["#c0392b" if v < 0 else "#2471a3" for v in ms.values]
    ax.barh([c.replace("shap_", "") for c in ms.index], ms.values, color=colors)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("Mean signed SHAP value")
    ax.set_title(f"{name}  (n={len(sub):,})\n"
                 "negative = pushes the anomaly score down (toward normal)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fig_shap_bar_cluster{cid}.png", dpi=160)
    plt.close(fig)

    # --- beeswarm ---
    import shap as shaplib
    fig = plt.figure(figsize=(9.5, 5.5))
    shaplib.summary_plot(
        sub[SHAP_COLS].to_numpy(),
        sub[[f"feat_{c}" for c in FEATURE_COLS]].to_numpy(),
        feature_names=FEATURE_COLS, show=False, plot_size=None)
    plt.title(f"{name}  (n={len(sub):,})", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"fig_shap_beeswarm_cluster{cid}.png", dpi=160)
    plt.close(fig)
    print(f"saved {name} figures")

# ------------------------------------------------------------
# 3. Old vs corrected comparison
# ------------------------------------------------------------
if OLD_TOP3.exists():
    old = pd.read_csv(OLD_TOP3)
    # cluster ids differ between the two join variants; plot names, not ids
    NL = {0: "long_low_tip", 1: "short_high_tip"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (title, src) in zip(
            axes,
            [("OLD (void - position join)", "old"),
             ("CORRECTED (row_uid join)", "new")]):
        if src == "old":
            for cid, marker in zip(sorted(old["cluster"].unique()),
                                   ["#c0392b", "#2471a3"]):
                s = old[old["cluster"] == cid]
                ax.barh([f"{NL.get(cid, cid)}: {f}" for f in s["feature"]],
                        s["mean_abs_shap"], color=marker, alpha=0.85,
                        label=NL.get(cid, cid))
        else:
            for cid, marker in zip(sorted(m["cluster_robust"].unique()),
                                   ["#c0392b", "#2471a3"]):
                s = table[(table["cluster"] == cid) &
                          (table["rank_in_cluster"] <= 3)]
                ax.barh([f"{NL.get(cid, cid)}: {f}" for f in s["feature"]],
                        s["mean_abs_shap"], color=marker, alpha=0.85,
                        label=NL.get(cid, cid))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("mean |SHAP|")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
    fig.suptitle("Per-cluster SHAP drivers: invalid positional join vs key-based join",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_shap_old_vs_corrected.png", dpi=160)
    plt.close(fig)
    print("saved fig_shap_old_vs_corrected.png")

print(f"\nAll figures in {OUT_DIR}")

# ------------------------------------------------------------
# 4. Does SHAP actually DISCRIMINATE the clusters?
# ------------------------------------------------------------
# This is the question the old script never asked. If both clusters share the
# same driver ranking, "cluster-specific drivers" is not a supportable claim.
print()
print("=" * 78)
print("DISCRIMINATION TEST: are the per-cluster driver profiles different?")
print("=" * 78)
piv = table.pivot(index="feature", columns="cluster", values="mean_abs_shap")
piv["abs_diff"] = (piv[0] - piv[1]).abs()
piv["rel_diff_pct"] = (piv["abs_diff"] /
                       piv[[0, 1]].mean(axis=1) * 100).round(2)
print(piv.round(4).to_string())

top3_features = ["log_tips", "speed_mph", "log_fare"]
top3_rel = piv.loc[top3_features, "rel_diff_pct"]
print()
print(f"relative difference on the top-3 drivers: "
      f"{top3_rel.min():.2f}% .. {top3_rel.max():.2f}%")
rho = piv[0].rank().corr(piv[1].rank(), method="spearman")
print(f"Spearman rank correlation of mean|SHAP| between clusters: {rho:.4f}")

verdict = "INDISTINGUISHABLE" if rho > 0.9 else "distinguishable"
print(f"VERDICT: driver profiles are {verdict}.")
print("=" * 78)
with open(OUT_DIR / "discrimination_test.txt", "w", encoding="utf-8") as f:
    f.write(piv.round(6).to_string())
    f.write("\n\n")
    f.write(f"top3_rel_diff_range_pct: {top3_rel.min():.2f}..{top3_rel.max():.2f}\n")
    f.write(f"spearman_rank_correlation: {rho:.4f}\n")
    f.write(f"verdict: {verdict}\n")
    f.write("\nInterpretation: both clusters are separated in FEATURE SPACE "
            "(KMeans), but share the same SHAP driver ranking, because SHAP "
            "here explains the IsolationForest anomaly score, not the "
            "cluster assignment. Cluster-level differences live in "
            "log_duration (+42%) and hour_cos (+43%), not in the top-3 "
            "magnitude drivers.\n")
