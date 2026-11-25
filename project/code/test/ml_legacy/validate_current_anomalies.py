# Location: project/code/test/ml_legacy/validate_current_anomalies.py | PROJECT_ROOT depth: 3
# (formerly project/code/ml/validate_current_anomalies.py - data validation, not part of the main pipeline)
# validate_current_anomalies.py
from pathlib import Path
import pandas as pd
import numpy as np

# .../project  (project/code/test/ml_legacy/x.py -> parents[3])
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC = PROJECT_ROOT / "data" / "output" / "out_of_use" / "_outputs" / "if_results_10pct_of_10pct" / "strict_anomalies"

df = pd.read_parquet(SRC)
print(f"Loaded {len(df):,} rows from {SRC.parent.name}/strict_anomalies\n")

tips = np.expm1(df["log_tips"])
fare = np.expm1(df["log_fare"])
miles = np.expm1(df["log_miles"])
dur_min = np.expm1(df["log_duration"])
recon = miles / (dur_min / 60.0)

print("=== Range check (vs sample_v2.py caps) ===")
print(f"tips      : min={tips.min():7.2f}  max={tips.max():8.2f}  >100 : {(tips > 100).sum():,}")
print(f"fare      : min={fare.min():7.2f}  max={fare.max():8.2f}  >250 : {(fare > 250).sum():,}")
print(f"miles     : min={miles.min():7.2f}  max={miles.max():8.2f}  >100 : {(miles > 100).sum():,}")
print(f"speed_mph : min={df['speed_mph'].min():7.2f}  max={df['speed_mph'].max():8.2f}  "
      f"outside[1,80]: {((df['speed_mph'] < 1) | (df['speed_mph'] > 80)).sum():,}")

print("\n=== Speed consistency ===")
print(f"corr(reconstructed, speed_mph) = {np.corrcoef(recon, df['speed_mph'])[0,1]:.4f}")
print(f"recon speed median = {recon.median():.2f} mph | recorded median = {df['speed_mph'].median():.2f} mph")

print("\n=== Zero-tip ===")
z = (df["log_tips"] == 0).sum()
print(f"zero-tip rows: {z:,} / {len(df):,} = {z/len(df):.2%}")