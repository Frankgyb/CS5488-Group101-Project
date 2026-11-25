"""
Location: project/code/test/ml_legacy/investigate_tips_field.py | PROJECT_ROOT depth: 3
(formerly project/code/ml/investigate_tips_field.py - data validation, not part of the main pipeline)
investigate_tips_field.py - Analyze the tip field in cluster 0 (abnormal_tip_ratio).
Goal: identify repeated values that suggest placeholder/default codes.
Output: top-30 most frequent tip values with context (fare, miles, count).
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Location: code/test/ml_legacy/investigate_tips_field.py -> 4 levels up = project/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WORK_DIR = PROJECT_ROOT / "data" / "output" / "anomaly_features"
SRC_PARQUET = WORK_DIR / "anomalies_enriched.parquet"

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
print(f"Reading {SRC_PARQUET.name}...")
df = pd.read_parquet(SRC_PARQUET)
print(f"Loaded {len(df)} rows, {df.shape[1]} cols")

# ------------------------------------------------------------
# 2. Filter cluster 0 (short_lowfare_no_tip)
# ------------------------------------------------------------
cluster0 = df[df["cluster_name"] == "short_lowfare_no_tip"].copy()
print(f"Cluster 0 size: {len(cluster0)}")

# Recover physical tips (same logic as upstream)
tips_raw = np.expm1(cluster0["log_tips"].clip(lower=0.0)).round(2)

# ------------------------------------------------------------
# 3. Value counts on tips
# ------------------------------------------------------------
vc = tips_raw.value_counts().reset_index()
vc.columns = ["tips_value", "count"]
vc = vc.sort_values("count", ascending=False).reset_index(drop=True)

print("\n=== Top-30 most frequent tip values (cluster 0) ===")
top30 = vc.head(30)
print(top30.to_string(index=False))

# ------------------------------------------------------------
# 4. For each top-30 value, compute summary stats of fare/miles
# ------------------------------------------------------------
print("\n=== Context for top-30 tip values ===")
rows_list = []
for val in top30["tips_value"]:
    mask = (tips_raw == val)
    sub = cluster0[mask]
    rows_list.append({
        "tips_value": val,
        "count": int(mask.sum()),
        "avg_fare": round(sub["fare_per_mile"].mean() * np.expm1(sub["log_fare"].clip(lower=0)).mean(), 2) if mask.sum() > 0 else None,
        "avg_miles": round(np.expm1(sub["log_miles"].clip(lower=0)).mean(), 2),
        "avg_fare_per_mile": round(sub["fare_per_mile"].mean(), 4),
        "min_fare": round(np.expm1(sub["log_fare"].clip(lower=0)).min(), 2),
        "max_fare": round(np.expm1(sub["log_fare"].clip(lower=0)).max(), 2),
    })

ctx = pd.DataFrame(rows_list)
print(ctx.to_string(index=False))

# ------------------------------------------------------------
# 5. Also check if these values appear in cluster 1 (cross-check)
# ------------------------------------------------------------
cluster1 = df[df["cluster_name"] == "long_detour_highfare"].copy()
tips1 = np.expm1(cluster1["log_tips"].clip(lower=0)).round(2)
vc1 = tips1.value_counts()

print("\n=== Cross-check: top-30 values also present in cluster 1? ===")
for val in top30["tips_value"]:
    cnt1 = vc1.get(val, 0)
    cnt0 = top30[top30["tips_value"] == val]["count"].values[0]
    print(f"  tips={val:8.2f}  cluster0_count={cnt0:5d}  cluster1_count={cnt1:5d}  "
          f"ratio_cluster1={cnt1/(cnt0+cnt1):.1%}" if cnt0+cnt1>0 else "")

# ------------------------------------------------------------
# 6. Save top-30 table
# ------------------------------------------------------------
out_path = WORK_DIR / "tips_top30_investigation.csv"
ctx.to_csv(out_path, index=False)
print(f"\nSaved context table to {out_path}")
print("Done.")