"""
Location: project/code/test/ml_legacy/check_fare_cap.py | PROJECT_ROOT depth: 3
(formerly project/code/ml/check_fare_cap.py - data validation, not part of the main pipeline)
check_fare_cap.py - Verify whether fare = 54.08 is a hard cap.
Reads the enriched parquet, inspects the fare distribution tail.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Location: code/test/ml_legacy/check_fare_cap.py -> 4 levels up = project/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WORK_DIR = PROJECT_ROOT / "data" / "output" / "anomaly_features"
df = pd.read_parquet(WORK_DIR / "anomalies_enriched.parquet")

fare = np.expm1(df["log_fare"].clip(lower=0.0))

print("=== fare distribution tail ===")
for q in [0.90, 0.95, 0.99, 0.995, 0.999, 1.0]:
    print(f"  p{q*100:5.1f} = {fare.quantile(q):.3f}")

print(f"\nmax = {fare.max():.3f}")
print(f"count == max : {(fare == fare.max()).sum()}  ({(fare == fare.max()).mean():.2%})")
print(f"count >= 50  : {(fare >= 50).sum()}  ({(fare >= 50).mean():.2%})")
print(f"count in [53.5, 54.08] : {((fare >= 53.5) & (fare <= 54.08)).sum()}")

# How many rows sit exactly at the cap, split by cluster?
at_cap = fare >= fare.max() - 0.01
print("\n=== rows at cap, by cluster ===")
print(df.loc[at_cap, "cluster_name"].value_counts().to_string())

# Among rows at cap, what is tip_ratio like?
print("\n=== tip_ratio for rows at cap ===")
sub = df.loc[at_cap, "tip_ratio"].dropna()
print(f"  n={len(sub)}  med={sub.median():.2f}  p90={sub.quantile(.9):.2f}  "
      f"share>2={(sub>2).mean():.1%}")

# Compare: rows NOT at cap
sub2 = df.loc[~at_cap, "tip_ratio"].dropna()
print(f"not at cap: n={len(sub2)}  med={sub2.median():.2f}  share>2={(sub2>2).mean():.1%}")