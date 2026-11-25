"""
Location: project/code/ml/analysis/05_cluster_money_summary.py | PROJECT_ROOT depth: 3
cluster_money_summary.py - Convert log features back to real money/units
and produce a per-cluster business summary for the report.

Reads the named clustered parquet (with cluster_name column),
reverses the log1p transforms, and prints + saves a summary table.

Output: data/output/anomaly_features/cluster_money_summary.csv  (NOTE 2026-11-22: the
        input parquet lives in out_of_use/_outputs/anomaly_features_deprecated/
        because 04's SRC was the legacy K=2 run, which has since been moved. This
        script is therefore a REPRODUCIBILITY entry point for a table whose MONEY
        figures are already published; the live, unit-audited equivalent is
        data/output/anomaly_features_v2/consistency_profile_v2.csv, keyed on
        cluster NAME. Do not add new work on top of this script.)
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. Paths (script lives in project/code/ml/)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../project
# DEPRECATED 2026-11-22: reads the legacy named parquet (no row_uid, legacy
# cluster ids). The MONEY figures it produced are correct and are reproduced
# verbatim by data/output/anomaly_features_v2/consistency_profile_v2.csv;
# the cluster IDS it carries are void. Prefer anomaly_features_v2/.
# 2026-11-22: 04 now writes the named parquet into the LIVE anomaly_features/, so
# read that copy; fall back to the archived duplicate for old checkouts.
SRC = (PROJECT_ROOT / "data" / "output" / "anomaly_features" /
       "anomalies_clustered_named.parquet")
if not SRC.exists():
    SRC = (PROJECT_ROOT / "data" / "output" / "out_of_use" / "_outputs" /
           "anomaly_features_deprecated" / "anomalies_clustered_named.parquet")
DST_DIR = PROJECT_ROOT / "data" / "output" / "anomaly_features"
DST_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. Load
# ------------------------------------------------------------
df = pd.read_parquet(SRC)
print(f"Loaded {len(df):,} rows from {SRC.name}")
print(f"Clusters: {df['cluster_name'].value_counts().to_dict()}\n")

# ------------------------------------------------------------
# 3. Reverse log1p transforms -> real values
#    (features were stored as log1p(x) = ln(1+x))
# ------------------------------------------------------------
df["miles"] = np.expm1(df["log_miles"])
df["fare"] = np.expm1(df["log_fare"])
df["tips"] = np.expm1(df["log_tips"])
df["duration_min"] = np.expm1(df["log_duration"])  # assuming duration was in minutes

# Real tip rate, guarded against zero/near-zero fares
df["tip_rate"] = np.where(df["fare"] > 0, df["tips"] / df["fare"], np.nan)

# Flags
df["is_zero_tip"] = df["tips"] < 0.005          # effectively zero after expm1
df["miles_capped"] = df["miles"] >= 99.5        # the 266 rows pinned at the 100 cap

# ------------------------------------------------------------
# 4. Per-cluster summary
# ------------------------------------------------------------
def pct(series):
    return 100.0 * series.mean()

summary = df.groupby("cluster_name").agg(
    n=("cluster_name", "size"),
    med_fare=("fare", "median"),
    med_tip=("tips", "median"),
    med_miles=("miles", "median"),
    med_duration_min=("duration_min", "median"),
    med_speed_mph=("speed_mph", "median"),
    mean_tip_rate=("tip_rate", "mean"),
    med_tip_rate=("tip_rate", "median"),
    zero_tip_pct=("is_zero_tip", pct),
    miles_capped_n=("miles_capped", "sum"),
).reset_index()

# Overall row for context
overall = pd.DataFrame([{
    "cluster_name": "ALL",
    "n": len(df),
    "med_fare": df["fare"].median(),
    "med_tip": df["tips"].median(),
    "med_miles": df["miles"].median(),
    "med_duration_min": df["duration_min"].median(),
    "med_speed_mph": df["speed_mph"].median(),
    "mean_tip_rate": df["tip_rate"].mean(),
    "med_tip_rate": df["tip_rate"].median(),
    "zero_tip_pct": pct(df["is_zero_tip"]),
    "miles_capped_n": int(df["miles_capped"].sum()),
}])

out = pd.concat([summary, overall], ignore_index=True)

# Round for display
disp = out.copy()
for c in ["med_fare", "med_tip", "med_miles", "med_duration_min", "med_speed_mph"]:
    disp[c] = disp[c].round(2)
for c in ["mean_tip_rate", "med_tip_rate", "zero_tip_pct"]:
    disp[c] = disp[c].round(2)

print("=== Per-cluster real-value summary ===")
print(disp.to_string(index=False))

# ------------------------------------------------------------
# 5. Save
# ------------------------------------------------------------
out_path = DST_DIR / "cluster_money_summary.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved {out_path}")

# ------------------------------------------------------------
# 6. Quick read-out for the report
# ------------------------------------------------------------
print("\n=== Report-ready highlights ===")
for _, r in summary.iterrows():
    print(
        f"{r['cluster_name']}: n={int(r['n']):,} | "
        f"median fare=${r['med_fare']:.2f} | median tip=${r['med_tip']:.2f} | "
        f"tip rate={r['mean_tip_rate']*100:.1f}% | "
        f"zero-tip={r['zero_tip_pct']:.1f}% | "
        f"median miles={r['med_miles']:.1f} | "
        f"median speed={r['med_speed_mph']:.1f} mph"
    )

