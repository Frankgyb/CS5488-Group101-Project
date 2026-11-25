"""
Location: project/code/test/ml_legacy/check_units.py | PROJECT_ROOT depth: 3
(formerly project/code/ml/check_units.py - data validation, not part of the main pipeline)
check_units.py - Diagnose the true unit of duration and miles
before building speed-based features.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Location: code/test/ml_legacy/check_units.py -> 4 levels up = project/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC = PROJECT_ROOT / "data" / "output" / "anomaly_clusters_rowkeyed" / "strict_anomalies_clustered"

df = pd.read_parquet(SRC)

print("=== Raw log feature stats ===")
for c in ["log_miles", "log_fare", "log_tips", "log_duration", "speed_mph"]:
    s = df[c]
    print(f"{c:14s} min={s.min():8.3f}  med={s.median():8.3f}  max={s.max():10.3f}")

print("\n=== Inverse-transformed (expm1) stats ===")
for c in ["log_miles", "log_fare", "log_tips", "log_duration"]:
    raw = np.expm1(df[c].clip(lower=0))
    print(f"{c:14s} min={raw.min():8.3f}  med={raw.median():8.3f}  "
          f"p95={raw.quantile(.95):9.2f}  max={raw.max():12.2f}")

print("\n=== Implied speed under 3 duration-unit hypotheses ===")
miles = np.expm1(df["log_miles"].clip(lower=0))
dur   = np.expm1(df["log_duration"].clip(lower=0))

for label, divisor in [("duration as seconds", 3600),
                       ("duration as minutes", 60),
                       ("duration as hours",   1)]:
    spd = miles / (dur / divisor)
    spd = spd.replace([np.inf, -np.inf], np.nan)
    ok = spd[(spd > 0) & (spd < 200)]
    print(f"{label:22s} median={ok.median():7.2f} mph   "
          f"p90={ok.quantile(.9):7.2f}   in[1,120]={((ok>=1)&(ok<=120)).mean():.1%}")

print("\n=== Reported speed_mph for reference ===")
s = df["speed_mph"]
print(f"min={s.min():.2f}  med={s.median():.2f}  max={s.max():.2f}  "
      f"in[1,120]={((s>=1)&(s<=120)).mean():.1%}")

print("\n=== Sample rows ===")
cols = ["log_miles", "log_fare", "log_tips", "log_duration", "speed_mph"]
print(df[cols].head(10).round(3).to_string())