# investigate_feature_issues.py
# Purpose: Diagnose three critical issues found in anomaly dataset:
# 1. Negative and uncapped tips (exceeding TIPS_CAP=100)
# 2. Speed inconsistency (reconstructed vs recorded)
# 3. Hidden fare cap at 54.08
#
# Location: project/code/test/investigate_feature_issues.py | PROJECT_ROOT depth: 2

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]      # .../project

# Live anomaly table (row-keyed, 13,577 rows). The 2% precision variant that this
# script used to read was archived under data/output/out_of_use/_outputs/.
ANOMALY_PATH = PROJECT_ROOT / "data" / "output" / "if_results_rowkeyed" / "strict_anomalies_rowkeyed"
anom = pd.read_parquet(ANOMALY_PATH)

print(f"Total records: {len(anom)}\n")

# ---- 1. Tips: negative values and over-cap analysis ----
print("=== Tips Anomaly Analysis ===")
original_tips = np.expm1(anom["log_tips"])
negative_tips = (original_tips < 0).sum()
over_cap_tips = (original_tips > 100).sum()
print(f"Negative tips records: {negative_tips} ({negative_tips/len(anom)*100:.2f}%)")
print(f"Tips exceeding 100: {over_cap_tips} ({over_cap_tips/len(anom)*100:.2f}%)")
print(f"Min tips: {original_tips.min():.2f}, Max tips: {original_tips.max():.2f}")
print(f"Median tips: {original_tips.median():.2f}\n")

print("Top 5 highest tips:")
top5 = anom.loc[original_tips.nlargest(5).index, ["log_miles","log_fare","log_tips","speed_mph","log_duration"]]
print(top5.to_string())
print()

# ---- 2. Speed inconsistency ----
print("=== Speed Inconsistency Diagnosis ===")
recon_speed = np.expm1(anom["log_miles"]) / (np.expm1(anom["log_duration"]) / 60.0)
speed_diff = anom["speed_mph"] - recon_speed
print(f"Speed difference (recorded - reconstructed):")
print(speed_diff.describe())
print(f"\nRecords with largest positive deviation:")
worst_pos = anom.loc[speed_diff.nlargest(5).index, ["log_miles","log_duration","speed_mph"]]
worst_pos["recon_speed"] = recon_speed.loc[worst_pos.index]
print(worst_pos.to_string())
print(f"\nRecords with largest negative deviation:")
worst_neg = anom.loc[speed_diff.nsmallest(5).index, ["log_miles","log_duration","speed_mph"]]
worst_neg["recon_speed"] = recon_speed.loc[worst_neg.index]
print(worst_neg.to_string())
print()

# ---- 3. Fare hidden cap at 54.08 ----
print("=== Fare Cap Investigation ===")
original_fare = np.expm1(anom["log_fare"])
cap_threshold = 54.08
is_capped = original_fare >= cap_threshold
print(f"Records at or above {cap_threshold}: {is_capped.sum()} ({is_capped.mean()*100:.2f}%)")
print(f"Fare quantiles: p50={original_fare.median():.2f}, p90={original_fare.quantile(0.9):.2f}, "
      f"p95={original_fare.quantile(0.95):.2f}, p99={original_fare.quantile(0.99):.2f}, "
      f"max={original_fare.max():.2f}")

capped_group = anom[is_capped]
uncapped_group = anom[~is_capped]
print(f"\nCapped group size: {len(capped_group)}, Uncapped group size: {len(uncapped_group)}")
print(f"Capped median speed_mph: {capped_group['speed_mph'].median():.2f}")
print(f"Uncapped median speed_mph: {uncapped_group['speed_mph'].median():.2f}")
print(f"Capped median log_miles: {capped_group['log_miles'].median():.4f}")
print(f"Uncapped median log_miles: {uncapped_group['log_miles'].median():.4f}")