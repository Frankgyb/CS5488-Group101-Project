"""
Location: project/code/ml/pipeline/04_rename_clusters.py | PROJECT_ROOT depth: 3
rename_clusters.py - Map numeric cluster ids to business names for the LEGACY
K=2 run, and materialise the named tables the plotting scripts consume.

Reads : data/output/anomaly_clusters_rowkeyed/   (03's LIVE output, 2026-11-22)
Writes: data/output/anomaly_features/            (live, consumed by 05 / 06 / viz)
        data/output/out_of_use/_outputs/anomaly_features_deprecated/  (copy, 2026-11-22)

Original files under anomaly_clusters_rowkeyed/ are never modified.

Cluster 0 -> short_high_tip    (short trip with unusually high tip)
Cluster 1 -> long_low_tip      (long trip with unusually low/no tip)

DEPRECATED as a source of truth (2026-11-22): the numbering above is the LEGACY
03 numbering and is the REVERSE of the authoritative robust clustering in
data/output/clustering_robustness/. Anything that quotes cluster IDs should read
data/output/anomaly_features_v2/ (cluster_robust + cluster_name) instead. The
tables written here are still valid for plotting because the colour/series logic
downstream keys on `cluster_name`, not on the id.
"""

from pathlib import Path
import pandas as pd

# ------------------------------------------------------------
# 1. Relative path setup (script at code/ml/)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../project

# LIVE source (2026-11-22): 03 writes to anomaly_clusters_rowkeyed/, which is the
# same folder 10_robust_clustering_rowkeyed.py and add_consistency_features_v2.py read.
SRC_DIR = PROJECT_ROOT / "data" / "output" / "anomaly_clusters_rowkeyed"
# LIVE destination: 06_plot_cluster_viz.py reads the named parquet from here.
DST_DIR = PROJECT_ROOT / "data" / "output" / "anomaly_features"
ARCHIVE_DIR = (PROJECT_ROOT / "data" / "output" / "out_of_use" / "_outputs" /
               "anomaly_features_deprecated")
DST_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

CLUSTER_MAP = {
    0: "short_high_tip",
    1: "long_low_tip",
}

FEATURE_COLS = [
    'log_miles', 'log_fare', 'log_tips', 'speed_mph', 'log_duration',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
]

# ------------------------------------------------------------
# 2. cluster_centers with business names
# ------------------------------------------------------------
centers = pd.read_csv(SRC_DIR / "cluster_centers.csv")
centers["cluster_name"] = centers["cluster"].map(CLUSTER_MAP)
centers_out = centers[["cluster", "cluster_name"] + FEATURE_COLS]
# 2026-11-22: archived on purpose. `cluster_centers.csv` holds StandardScaler-space
# values under the original feature names (e.g. speed_mph = 3.433, not mph), and a
# mislabelled table in the live folder was repeatedly read as physical units.
# Money/behavioural numbers come from 05_cluster_money_summary.py instead.
STALE_DIR = (PROJECT_ROOT / "data" / "output" / "out_of_use" / "_outputs" /
             "anomaly_features_stale")
STALE_DIR.mkdir(parents=True, exist_ok=True)
centers_out.to_csv(STALE_DIR / "cluster_centers_named.csv", index=False)
centers_out.to_csv(ARCHIVE_DIR / "cluster_centers_named.csv", index=False)
print("Saved cluster_centers_named.csv  -> out_of_use/_outputs/anomaly_features_stale/ (scaler space, do not quote units)")
print(centers_out[["cluster", "cluster_name"]].to_string(index=False))

# ------------------------------------------------------------
# 3. cluster_sizes with business names
# ------------------------------------------------------------
sizes = pd.read_csv(SRC_DIR / "cluster_sizes.csv")
sizes["cluster_name"] = sizes["cluster"].map(CLUSTER_MAP)
sizes_out = sizes[["cluster", "cluster_name", "count", "ratio"]]
sizes_out.to_csv(DST_DIR / "cluster_sizes_named.csv", index=False)
sizes_out.to_csv(ARCHIVE_DIR / "cluster_sizes_named.csv", index=False)
print("\nSaved cluster_sizes_named.csv")
print(sizes_out.to_string(index=False))

# ------------------------------------------------------------
# 4. Copy clustered parquet and add cluster_name column
# ------------------------------------------------------------
df = pd.read_parquet(SRC_DIR / "strict_anomalies_clustered")  # Spark directory, no extension
df["cluster_name"] = df["cluster"].map(CLUSTER_MAP)
out_parquet = DST_DIR / "anomalies_clustered_named.parquet"
df.to_parquet(out_parquet, index=False)
print(f"\nSaved {out_parquet}  ({len(df)} rows, {df.shape[1]} cols)")
df.to_parquet(ARCHIVE_DIR / "anomalies_clustered_named.parquet", index=False)

# ------------------------------------------------------------
# 5. Persona summary table for the report
# ------------------------------------------------------------
c = centers.set_index("cluster")
rows = []
for cid, name in CLUSTER_MAP.items():
    r = c.loc[cid]
    rows.append({
        "cluster": cid,
        "cluster_name": name,
        "profile": "short trip, low fare, low speed, high tip"
                   if cid == 0 else
                   "long distance, high fare, high duration, high speed, low/no tip",
        "suspected_pattern": "short trip with unusually high tip"
                            if cid == 0 else
                            "long trip with unusually low or no tip",
        # DEPRECATED 2026-11-22: these literals came from the old SHAP table that
        # joined SHAP rows to cluster labels by row POSITION instead of a key, so
        # the per-cluster driver rankings were random. Both clusters actually rank
        # log_tips > speed_mph > log_fare (Spearman rho = 0.983 => indistinguishable).
        # Read authoritative values from data/output/shap_final/shap_top_drivers_final.csv.
        # See the corrected outputs in data/output/shap_final/.
        "top_shap_driver": "STALE - see shap_final/shap_top_drivers_final.csv",
        "time_features_role": "near zero, not time-driven",
        "log_miles_z": round(r["log_miles"], 2),
        "log_fare_z": round(r["log_fare"], 2),
        "log_tips_z": round(r["log_tips"], 2),
        "speed_mph_z": round(r["speed_mph"], 2),
        "log_duration_z": round(r["log_duration"], 2),
    })

persona = pd.DataFrame(rows)
# 2026-11-22: also archived. `top_shap_driver` is a STALE placeholder and the other
# columns are raw personality prose; the authoritative driver ranking lives in
# data/output/shap_final/shap_top_drivers_final.csv.
persona.to_csv(STALE_DIR / "cluster_persona.csv", index=False)
persona.to_csv(ARCHIVE_DIR / "cluster_persona.csv", index=False)
print("\nSaved cluster_persona.csv  -> out_of_use/_outputs/anomaly_features_stale/")
print(persona[["cluster", "cluster_name", "suspected_pattern"]].to_string(index=False))

print("\nDone. Original files in anomaly_clusters_rowkeyed/ untouched.")
print(f"Live copy  : {DST_DIR}")
print(f"Archive    : {ARCHIVE_DIR}")