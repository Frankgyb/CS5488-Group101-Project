# validate_feature_consistency.py
# Purpose: Verify that log-transformed features can be correctly inverse-transformed,
# that speed reconstruction is consistent with the recorded speed_mph,
# and that zero-tip identification works as expected.
# Author: Your Name
# Date: 2026-11-19
# Location: project/code/test/validate_feature_consistency.py | PROJECT_ROOT depth: 2

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]      # .../project

# ------------------------------------------------------------------
# 1. Load anomaly dataset (strict anomalies from Isolation Forest)
#    Live row-keyed table; the 2% precision variant is archived under
#    data/output/out_of_use/_outputs/.
# ------------------------------------------------------------------
ANOMALY_PATH = PROJECT_ROOT / "data" / "output" / "if_results_rowkeyed" / "strict_anomalies_rowkeyed"

anom = pd.read_parquet(ANOMALY_PATH)
print(f"Loaded {len(anom)} anomaly records.\n")

# ------------------------------------------------------------------
# 2. Inverse transform check: expm1(log1p(x)) should recover original x
# ------------------------------------------------------------------
print("=== Inverse Transform Check (expm1) ===")
for col, raw_name in [
    ("log_miles", "miles"),
    ("log_fare", "fare"),
    ("log_tips", "tips"),
    ("log_duration", "minutes")
]:
    orig = np.expm1(anom[col])  # inverse of log1p
    print(f"{raw_name:10s}: median={orig.median():8.2f}, "
          f"max={orig.max():12.2f}, p99={orig.quantile(0.99):8.2f}")

print()  # blank line

# ------------------------------------------------------------------
# 3. Speed consistency: reconstruct speed from log_miles & log_duration
#    Expected formula: speed_mph = miles / (minutes / 60)
# ------------------------------------------------------------------
print("=== Speed Reconstruction Consistency ===")
reconstructed_speed = np.expm1(anom["log_miles"]) / (np.expm1(anom["log_duration"]) / 60.0)
correlation = np.corrcoef(reconstructed_speed, anom["speed_mph"])[0, 1]
print(f"Correlation between reconstructed speed and speed_mph: {correlation:.6f}")
if correlation > 0.999:
    print("OK - perfect consistency: speed_mph is indeed derived from miles/duration.")
else:
    print("WARNING - inconsistency detected, investigate further.")

print()  # blank line

# ------------------------------------------------------------------
# 4. Zero-tip detection: log_tips ≈ log1p(0) = 0
# ------------------------------------------------------------------
print("=== Zero-Tip Detection ===")
zero_tip_mask = np.abs(anom["log_tips"]) < 1e-6   # tolerance for floating point
zero_tip_count = zero_tip_mask.sum()
zero_tip_share = zero_tip_mask.mean()
print(f"Zero-tip records: {zero_tip_count} / {len(anom)} = {zero_tip_share:.2%}")

# Also show non-zero tip records for comparison
non_zero = anom[~zero_tip_mask]
if len(non_zero) > 0:
    print(f"Non-zero tip records: {len(non_zero)}")
    print(f"  log_tips range: [{non_zero['log_tips'].min():.4f}, {non_zero['log_tips'].max():.4f}]")
    print(f"  Original tips range: [{np.expm1(non_zero['log_tips']).min():.2f}, {np.expm1(non_zero['log_tips']).max():.2f}]")