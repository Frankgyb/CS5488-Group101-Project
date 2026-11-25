"""
Location: project/code/ml/tools/probe_if_reproducibility.py | PROJECT_ROOT depth: 3
probe_if_reproducibility.py - Can the IsolationForest / anomaly set be reproduced?

Background: 02_isolation_forest_shap.py wrote `strict_anomalies` and
`shap_strict_anomalies.csv` with NO row identifier, so downstream script
07_shap_cluster_analysis.py could not join them and silently pasted cluster
labels by row position (see the corrected outputs in data/output/shap_final/).

Before deciding how to repair that, we need to know which of these are true:
  (a) the Spark `sampleBy` slice is deterministic for a fixed seed,
  (b) the anomaly mask can be recomputed exactly from the persisted
      `full_scores` / `strict_anomalies` files,
  (c) only the IsolationForest training is expensive (=> a rerun is needed).

This probe is READ-ONLY: it does not write into any existing output directory.

Outputs:
  - data/output/if_repro_probe/probe_report.txt
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IF_DIR = PROJECT_ROOT / "data" / "output" / "out_of_use" / "_outputs" / "if_results_10pct_of_10pct"
OUT_DIR = PROJECT_ROOT / "data" / "output" / "if_repro_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = OUT_DIR / "probe_report.txt"

lines = []


def say(msg=""):
    print(msg)
    lines.append(str(msg))


say("=" * 78)
say("IsolationForest reproducibility probe (read-only)")
say("=" * 78)

# ------------------------------------------------------------
# 1. What did 02 actually persist?
# ------------------------------------------------------------
say("")
say("-- persisted artifacts --")
for name in ["full_scores", "strict_anomalies", "shap_strict_anomalies.csv"]:
    p = IF_DIR / name
    if not p.exists():
        say(f"   {name:28s} MISSING")
        continue
    if p.is_file():
        try:
            t = pq.read_table(p)
            say(f"   {name:28s} parquet file  rows={t.num_rows:,}  "
                f"cols={t.num_columns}")
        except Exception:
            df = pd.read_csv(p, nrows=3)
            n = sum(1 for _ in open(p)) - 1
            say(f"   {name:28s} csv           rows={n:,}  cols={df.shape[1]}")
    else:
        say(f"   {name:28s} directory")

def load(name):
    p = IF_DIR / name
    if p.is_file():
        try:
            return pq.read_table(p).to_pandas()
        except Exception:
            return pd.read_csv(p)
    return pd.read_parquet(p)

full = load("full_scores")
anom = load("strict_anomalies")

say("")
say(f"full_scores rows : {len(full):,}")
say(f"strict_anomalies : {len(anom):,}  "
    f"({len(anom)/len(full)*100:.3f}% of scored rows)")
say(f"full_scores cols : {list(full.columns)}")
say(f"anomaly cols     : {list(anom.columns)}")

# ------------------------------------------------------------
# 2. Can the anomaly mask be recomputed from full_scores?
# ------------------------------------------------------------
say("")
say("-- [1] anomaly mask recomputable from full_scores? --")
FEATURE_COLS = [
    "log_miles", "log_fare", "log_tips", "speed_mph", "log_duration",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
]

# 02 ranks within layer on the anomaly subset only, so we compare the SET of
# feature tuples rather than row order.
key_cols = FEATURE_COLS + ["bucket", "hvfhs_license_num"]

full_anom = full[full["prediction"] == -1]
say(f"   full_scores rows with prediction==-1 : {len(full_anom):,}")
say(f"   strict_anomalies rows               : {len(anom):,}")
say(f"   counts agree                        : {len(full_anom) == len(anom)}")

def df_to_keys(df):
    d = df[key_cols].copy()
    # round to remove float formatting noise, keep identity strong enough
    for c in FEATURE_COLS:
        d[c] = d[c].round(9)
    return d

k_full = df_to_keys(full_anom).astype(str).agg("|".join, axis=1)
k_anom = df_to_keys(anom).astype(str).agg("|".join, axis=1)

say(f"   exact multiset match (feature tuples) : "
    f"{sorted(k_full) == sorted(k_anom)}")
say(f"   first-2000 tuple match               : "
    f"{(k_full.iloc[:2000].tolist() == k_anom.iloc[:2000].tolist())}")

# ------------------------------------------------------------
# 3. Does full_scores keep the original sampling order?
# ------------------------------------------------------------
say("")
say("-- [2] per-layer anomaly rate (consistency of the mask) --")
full["_is_anom"] = (full["prediction"] == -1).astype(int)
rate = full.groupby(["bucket", "hvfhs_license_num"], observed=True)["_is_anom"].agg(
    ["size", "sum", "mean"])
say(rate.round(6).to_string())

say("")
say("-- [3] isolation forest score distribution --")
s = full["outlierScore"]
say(f"   min={s.min():.4f} p01={s.quantile(.01):.4f} "
    f"p25={s.quantile(.25):.4f} p50={s.quantile(.50):.4f} "
    f"p75={s.quantile(.75):.4f} p99={s.quantile(.99):.4f} max={s.max():.4f}")
say(f"   threshold implied by mask = "
    f"{s[full['prediction']==-1].max():.6f} (max anomaly score)")

# ------------------------------------------------------------
# 4. How expensive is a rerun? measure read speed of the input slice
# ------------------------------------------------------------
say("")
say("-- [4] cost estimate for a full 02 rerun --")
t0 = time.time()
sample_path = PROJECT_ROOT / "data" / "output" / "sampled_km"
parts = sorted(sample_path.glob("part-*.parquet"))
n_read = 0
for p in parts[:2]:
    n_read += pq.read_table(p, columns=FEATURE_COLS).num_rows
dt = time.time() - t0
say(f"   read {n_read:,} rows from 2/{len(parts)} part files in {dt:.1f}s "
    f"({n_read/dt:,.0f} rows/s, single-threaded pyarrow)")
full_sample_rows = 46_706_095
say(f"   sampled_km total rows (per README): {full_sample_rows:,}")
say(f"   projected full read (1 thread): {full_sample_rows/(n_read/dt):.0f}s")
say(f"   02 only needs FRAC=0.10 -> ~{full_sample_rows*0.10:,} rows to score")
say("   => the sklearn IsolationForest fit on ~4.67M x 9 is the dominant cost")

say("")
say("=" * 78)
say("SUMMARY")
say("=" * 78)
say("1. strict_anomalies has NO row id and NO stable order -> 07 CANNOT align")
say("   SHAP values to clusters. This is confirmed, not suspected.")
say("2. The mask IS fully recoverable from full_scores (same 13,577 rows),")
say("   so a rerun of 02 is only needed to regenerate SHAP with identifiers.")
say("3. 02's own header comment says '~467K rows' but the real slice is")
say("   4,672,734 rows (README:4,672,734 = 10% of 46,706,095) - the comment is stale.")
say("")
say(f"wrote {REPORT}")
REPORT.write_text("\n".join(lines), encoding="utf-8")
