"""
Location: project/code/ml/analysis/add_consistency_features.py | PROJECT_ROOT depth: 3
add_consistency_features.py (v3) - Unit-corrected consistency features.
Findings from check_units.py (now archived at project/code/test/ml_legacy/check_units.py):
  - log_duration unit is HOURS (only hypothesis yielding sane speeds)
  - speed_mph is already a standardized score, NOT physical mph
  - log_miles/log_fare/log_tips are log1p-transformed raw values
Therefore: build physical speed from miles/hours; keep standardized
speed_mph as a separate dimension; do NOT subtract across units.
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../project
WORK_DIR = PROJECT_ROOT / "data" / "output" / "anomaly_features"
WORK_DIR.mkdir(parents=True, exist_ok=True)

SRC_PARQUET = WORK_DIR / "anomalies_clustered_named.parquet"
DST_PARQUET = WORK_DIR / "anomalies_enriched.parquet"

# ------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------
print(f"Reading {SRC_PARQUET.name}...")
df = pd.read_parquet(SRC_PARQUET)
print(f"Loaded {len(df)} rows, {df.shape[1]} cols")

# ------------------------------------------------------------
# 2. Clip log features, recover physical scale
# ------------------------------------------------------------
for c in ["log_miles", "log_fare", "log_tips", "log_duration"]:
    df[c] = df[c].clip(lower=0.0)

miles  = np.expm1(df["log_miles"])      # miles
fare   = np.expm1(df["log_fare"])       # currency units
tips   = np.expm1(df["log_tips"])       # currency units
dur_hr = np.expm1(df["log_duration"])   # HOURS (confirmed by check_units)

# ------------------------------------------------------------
# 3. Physical consistency features
# ------------------------------------------------------------
# True average speed in mph, from distance / time
df["true_speed_mph"] = np.where(dur_hr > 0, miles / dur_hr, np.nan)

# Hours spent per mile (inverse speed, more stable for ratios)
df["hr_per_mile"] = np.where(miles > 0, dur_hr / miles, np.nan)

# Fare intensity
df["fare_per_mile"] = np.where(miles > 0, fare / miles, np.nan)

# Tip generosity
df["tip_ratio"] = np.where(fare > 0, tips / fare, np.nan)

# Fare per hour (revenue rate)
df["fare_per_hr"] = np.where(dur_hr > 0, fare / dur_hr, np.nan)

# ------------------------------------------------------------
# 4. Sanitize infinities
# ------------------------------------------------------------
NEW_COLS = ["true_speed_mph", "hr_per_mile", "fare_per_mile", "tip_ratio", "fare_per_hr"]
for c in NEW_COLS:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)

print("\nFeature ranges (unit-corrected):")
for c in NEW_COLS:
    s = df[c].dropna()
    s = s[(s >= 0) & (s < 1e4)]   # display-friendly view
    print(f"  {c:16s} med={s.median():8.2f}  p90={s.quantile(.9):8.2f}  "
          f"max={s.max():10.2f}  nulls={int(df[c].isna().sum())}")

# ------------------------------------------------------------
# 5. Winsorize for profiling (kill the 419-mph-style outliers)
# ------------------------------------------------------------
def wins(s, lo=0.01, hi=0.99):
    return s.clip(s.quantile(lo), s.quantile(hi))

print("\nCluster profile (winsorized 1%-99%):")
agg = (df.assign(_spd=wins(df["true_speed_mph"].fillna(np.nan)),
                 _fpm=wins(df["fare_per_mile"].fillna(np.nan)),
                 _tr =wins(df["tip_ratio"].fillna(np.nan)),
                 _hpm=wins(df["hr_per_mile"].fillna(np.nan)))
         .groupby("cluster_name")
         .agg(n=("cluster", "size"),
              spd_med=("_spd", "median"),
              spd_p90=("_spd", lambda x: x.quantile(.9)),
              hr_per_mile_med=("_hpm", "median"),
              fare_per_mile_med=("_fpm", "median"),
              fare_per_mile_p90=("_fpm", lambda x: x.quantile(.9)),
              tip_ratio_med=("_tr", "median"),
              tip_ratio_p90=("_tr", lambda x: x.quantile(.9)))
         .round(3).reset_index())

print(agg.to_string(index=False))
agg.to_csv(WORK_DIR / "consistency_profile.csv", index=False)

# ------------------------------------------------------------
# 6. Save
# ------------------------------------------------------------
df.to_parquet(DST_PARQUET, index=False)
print(f"\nSaved {DST_PARQUET} ({df.shape[1]} cols)")
print("Done.")