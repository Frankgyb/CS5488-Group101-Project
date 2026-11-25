"""
Location: project/code/test/ml_legacy/validate_tip_anomaly.py | PROJECT_ROOT depth: 3
(formerly project/code/ml/validate_tip_anomaly.py - data validation, not part of the main pipeline)
validate_tip_anomaly.py - Two validations on the enriched anomaly data.

1. tip_ratio sanity check: pull the Top-20 highest tip_ratio rows back to
   physical scale (fare / tips in currency) to judge whether extreme ratios
   are billing faults or genuine behaviour.

2. fare_per_mile by mileage band: median / p25 / p75 / p90 per band, so a
   defensible detour rule can be drawn from the distribution rather than
   from a guess.

Reads : data/output/out_of_use/_outputs/anomaly_features_deprecated/anomalies_enriched.parquet
Writes: data/output/out_of_use/_outputs/anomaly_features_deprecated/tip_ratio_top20.csv
        data/output/anomaly_features/fare_per_mile_overall.csv
        data/output/out_of_use/_outputs/anomaly_features_deprecated/fare_per_mile_by_cluster.csv
Original files are never modified.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. Relative path setup (script at code/ml/)
# ------------------------------------------------------------
# Location: code/test/ml_legacy/validate_tip_anomaly.py -> 4 levels up = project/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WORK_DIR = PROJECT_ROOT / "data" / "output" / "anomaly_features"
# DEPRECATED 2026-11-22: its input parquet was moved to out_of_use/ (no row_uid,
# legacy cluster ids). Kept for provenance; superseded by anomaly_features_v2/.
SRC_PARQUET = (PROJECT_ROOT / "data" / "output" / "out_of_use" / "_outputs" /
               "anomaly_features_deprecated" / "anomalies_enriched.parquet")

# ------------------------------------------------------------
# 2. Load enriched data
# ------------------------------------------------------------
print(f"Reading {SRC_PARQUET.name}...")
df = pd.read_parquet(SRC_PARQUET)
print(f"Loaded {len(df)} rows, {df.shape[1]} cols")

# Physical scale (clip log features at 0 first, same convention as upstream)
miles = np.expm1(df["log_miles"].clip(lower=0.0))
fare = np.expm1(df["log_fare"].clip(lower=0.0))
tips = np.expm1(df["log_tips"].clip(lower=0.0))
dur_hr = np.expm1(df["log_duration"].clip(lower=0.0))

audit = pd.DataFrame({
    "cluster_name": df["cluster_name"],
    "miles": miles.round(2),
    "fare": fare.round(2),
    "tips": tips.round(2),
    "dur_hr": dur_hr.round(2),
    "tip_ratio": df["tip_ratio"],
    "fare_per_mile": df["fare_per_mile"],
})

# ------------------------------------------------------------
# 3. Validation A - tip_ratio extremes back to physical scale
# ------------------------------------------------------------
top20 = (audit.dropna(subset=["tip_ratio"])
              .sort_values("tip_ratio", ascending=False)
              .head(20)
              .reset_index(drop=True))

print("\n=== tip_ratio Top-20 (physical scale) ===")
print(top20.to_string(index=False))

top20.to_csv(WORK_DIR / "tip_ratio_top20.csv", index=False)
print(f"\nSaved {WORK_DIR / 'tip_ratio_top20.csv'}")

# How concentrated are high tip_ratio values?
tr = audit["tip_ratio"].dropna()
print("\n=== tip_ratio distribution ===")
for q in [0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f"  p{int(q*100):2d} = {tr.quantile(q):.3f}")
print(f"  share with tip_ratio > 3 : {(tr > 3).mean():.1%}")
print(f"  share with tip_ratio > 5 : {(tr > 5).mean():.1%}")
print(f"  share with tip_ratio > 10: {(tr > 10).mean():.1%}")

# Within the short cluster only, how many rows have fare < 5?
short = audit[audit["cluster_name"] == "short_lowfare_no_tip"]
print(f"\nshort_lowfare_no_tip: n={len(short)}")
print(f"  fare < 5  : {(short['fare'] < 5).mean():.1%}")
print(f"  fare < 10 : {(short['fare'] < 10).mean():.1%}")
print(f"  tips > fare * 2 : {(short['tips'] > short['fare'] * 2).mean():.1%}")

# ------------------------------------------------------------
# 4. Validation B - fare_per_mile by mileage band
# ------------------------------------------------------------
bins = [0, 5, 10, 20, 40, 60, 80, np.inf]
labels = ["0-5", "5-10", "10-20", "20-40", "40-60", "60-80", "80+"]
audit["mile_band"] = pd.cut(audit["miles"], bins=bins, labels=labels, right=False)

# Overall distribution (all anomalies)
overall = (audit.dropna(subset=["fare_per_mile", "mile_band"])
                .groupby("mile_band", observed=True)
                .agg(
                    n=("fare_per_mile", "size"),
                    fpm_med=("fare_per_mile", "median"),
                    fpm_p25=("fare_per_mile", lambda x: x.quantile(0.25)),
                    fpm_p75=("fare_per_mile", lambda x: x.quantile(0.75)),
                    fpm_p90=("fare_per_mile", lambda x: x.quantile(0.90)),
                    fare_med=("fare", "median"),
                )
                .round(3)
                .reset_index())

print("\n=== fare_per_mile by mileage band (all anomalies) ===")
print(overall.to_string(index=False))

# Per-cluster distribution
by_cluster = (audit.dropna(subset=["fare_per_mile", "mile_band"])
                   .groupby(["cluster_name", "mile_band"], observed=True)
                   .agg(
                       n=("fare_per_mile", "size"),
                       fpm_med=("fare_per_mile", "median"),
                       fpm_p10=("fare_per_mile", lambda x: x.quantile(0.10)),
                       fpm_p90=("fare_per_mile", lambda x: x.quantile(0.90)),
                   )
                   .round(3)
                   .reset_index())

print("\n=== fare_per_mile by band x cluster ===")
print(by_cluster.to_string(index=False))

# Save both
overall.to_csv(WORK_DIR / "fare_per_mile_overall.csv", index=False)
by_cluster.to_csv(WORK_DIR / "fare_per_mile_by_cluster.csv", index=False)
print(f"\nSaved overall to {WORK_DIR / 'fare_per_mile_overall.csv'}")
print(f"Saved by_cluster to {WORK_DIR / 'fare_per_mile_by_cluster.csv'}")

# ------------------------------------------------------------
# 5. Candidate rule hit-test
# ------------------------------------------------------------
print("\n=== Candidate detour rule hit test ===")
for thr in [0.4, 0.5, 0.6, 0.7]:
    for mmin in [20, 40]:
        mask = (audit["miles"] > mmin) & (audit["fare_per_mile"] < thr)
        hit = int(mask.sum())
        if hit == 0:
            continue
        long_share = (audit.loc[mask, "cluster_name"] == "long_detour_highfare").mean()
        print(f"  miles > {mmin:2d} & fare_per_mile < {thr:.1f}  ->  "
              f"hit={hit:5d}  ({hit/len(audit):5.1%})  long_cluster_share={long_share:.1%}")

print("\nDone.")