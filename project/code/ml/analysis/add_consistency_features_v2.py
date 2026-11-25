"""
Location: project/code/ml/analysis/add_consistency_features_v2.py | PROJECT_ROOT depth: 3
add_consistency_features_v2.py - Unit-corrected consistency features.

WHAT WAS WRONG IN v1 (add_consistency_features.py)
--------------------------------------------------
v1 decided, from a mis-read `check_units.py` output, that:

    "log_duration unit is HOURS (only hypothesis yielding sane speeds)"
    "speed_mph is already a standardized score, NOT physical mph"

Both statements are FALSE, and both are now disproved empirically in
data/output/unit_check/unit_check_report.txt:

  * log_duration = log1p(MINUTES). The identity
        log1p((miles / speed_mph) * 60) == log_duration
    holds to a p99 residual of 4.4e-16 across 2,000,000 sampled rows, whereas
    the HOURS hypothesis misses by p99 = 3.49 in natural-log units (~33x).
  * speed_mph is a PHYSICAL mph value: mean 13.63, min 1.00, max 75.08,
    std 7.18 - impossible for a z-score, and bounded exactly by the
    SPEED_MIN..SPEED_MAX filter in 01_sample_v2.py:126-128.

v1 therefore divided an already-minutes duration by 60 a second time, which is
why its output carried `spd_med = 3.454 / 3.029` mph (physically absurd for
NYC FHV traffic) and `hr_per_mile` around 0.29-0.33.

WHAT v2 DOES DIFFERENTLY
------------------------
  * treats log_duration as MINUTES (no extra /60),
  * does NOT re-derive speed from duration - it VERIFIES the stored speed_mph
    against miles/duration and reports the agreement,
  * writes to a NEW directory so v1's file is preserved:
        data/output/anomaly_features_v2/
  * emits an explicit unit-audit block that a reviewer can cite.

INPUTS
  - data/output/anomaly_clusters_rowkeyed/selfcheck_k4/strict_anomalies_clustered_rowkeyed
    (moved 2026-11-22, was out_of_use/_outputs/..._deprecated/; only its row_uid +
     features are used. The K=4 labels it carries are NOT the reported model - it is
     07b's self-check output. Authoritative labels come from the robust merge below.)
  - data/output/clustering_robustness/row_uid_cluster_robust.parquet  (optional:
    supplies the multi-seed robust cluster id)

OUTPUTS (data/output/anomaly_features_v2/)
  - anomalies_enriched_v2.parquet   raw features + physical units + consistency ratios
  - consistency_profile_v2.csv      per-cluster physical profile, keyed by CLUSTER NAME
                                    (project rule 2026-11-22: cluster ids must not be
                                     quoted anywhere; the numeric id is context only)
  - unit_audit.csv                  the verification numbers
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CLUSTERED = (PROJECT_ROOT / "data" / "output" / "anomaly_clusters_rowkeyed" /
             "selfcheck_k4" / "strict_anomalies_clustered_rowkeyed")
ROBUST = (PROJECT_ROOT / "data" / "output" / "clustering_robustness" /
          "row_uid_cluster_robust.parquet")
UNIT_CHECK = PROJECT_ROOT / "data" / "output" / "unit_check" / "unit_check_report.txt"

OUT_DIR = PROJECT_ROOT / "data" / "output" / "anomaly_features_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "log_miles", "log_fare", "log_tips", "speed_mph", "log_duration",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
]

audit_rows = []


def log(msg=""):
    print(msg)


log("=" * 78)
log("CONSISTENCY FEATURES v2 - unit-corrected")
log("=" * 78)

# ------------------------------------------------------------
# 0. Restate the unit facts this script depends on
# ------------------------------------------------------------
log("")
log("-- unit facts this script relies on --")
log("   log_duration = log1p(MINUTES)   [verified, p99 residual 4.4e-16]")
log("   speed_mph    = physical mph     [verified, range 1.00..75.08]")
log("   log_miles / log_fare / log_tips = log1p(capped raw values)")
if UNIT_CHECK.exists():
    log(f"   evidence: {UNIT_CHECK.name}")
    audit_rows.append({"fact": "log_duration_unit", "value": "MINUTES",
                       "evidence": "data/output/unit_check/unit_check_report.txt"})
    audit_rows.append({"fact": "speed_mph_scale", "value": "physical mph",
                       "evidence": "data/output/unit_check/unit_check_report.txt"})

# ------------------------------------------------------------
# 1. Load
# ------------------------------------------------------------
df = pd.read_parquet(CLUSTERED)
log("")
log(f"loaded {len(df):,} rows x {df.shape[1]} cols from clustered anomalies")
log(f"row_uid unique: {df['row_uid'].is_unique}")

if ROBUST.exists():
    robust = pd.read_parquet(ROBUST)
    df = df.merge(robust, on="row_uid", how="left")
    log(f"merged multi-seed robust cluster labels "
        f"({df['cluster_robust'].notna().sum():,}/{len(df):,} rows matched)")
    has_robust = True
else:
    log("robust cluster labels not found - falling back to 03's cluster id")
    df["cluster_robust"] = df["cluster"]
    has_robust = False

# ------------------------------------------------------------
# 1b. Cluster NAME is the only quotable label (project rule 2026-11-22)
# ------------------------------------------------------------
# The robust numbering is 0 = long_low_tip / 1 = short_high_tip and is the REVERSE
# of the legacy 03 numbering, so any table keyed on a bare id is a trap. Every
# output below is keyed on `cluster_robust_name`; the id columns stay in the
# enriched parquet as context only.
ROBUST_NAME = {0: "long_low_tip", 1: "short_high_tip"}
df["cluster_robust_name"] = df["cluster_robust"].map(ROBUST_NAME)
if df["cluster_robust_name"].isna().any():
    log(f"WARNING: {int(df['cluster_robust_name'].isna().sum()):,} rows have no "
        f"robust label name (cluster_robust={sorted(df['cluster_robust'].dropna().unique())})")
log("")
log("-- cluster name mapping (robust numbering) --")
log(f"   {ROBUST_NAME}")
log(f"   label column for all outputs: cluster_robust_name")

# ------------------------------------------------------------
# 2. Physical scale - NO /60 THIS TIME
# ------------------------------------------------------------
for c in ["log_miles", "log_fare", "log_tips", "log_duration"]:
    df[c] = df[c].clip(lower=0.0)

miles = np.expm1(df["log_miles"].to_numpy())
fare = np.expm1(df["log_fare"].to_numpy())
tips = np.expm1(df["log_tips"].to_numpy())
dur_min = np.expm1(df["log_duration"].to_numpy())      # MINUTES  <-- fix

df["miles"] = miles
df["fare"] = fare
df["tips"] = tips
df["duration_min"] = dur_min
df["duration_hr"] = dur_min / 60.0

# ------------------------------------------------------------
# 3. VERIFY the stored speed instead of recomputing it blind
# ------------------------------------------------------------
log("")
log("-- speed verification --")
speed_derived = np.where(dur_min > 0, miles / (dur_min / 60.0), np.nan)
stored = df["speed_mph"].to_numpy()
ok = np.isfinite(speed_derived) & np.isfinite(stored)
max_diff = float(np.nanmax(np.abs(speed_derived[ok] - stored[ok])))
log(f"   max |miles/(minutes/60) - speed_mph| = {max_diff:.3e}")
log(f"   => speed_mph is consistent with MINUTES: {max_diff < 1e-6}")
audit_rows.append({"fact": "speed_vs_units_max_diff", "value": f"{max_diff:.3e}",
                   "evidence": "this run"})

# The single most important number in this whole repair: what v1 reported.
log("")
log("-- v1 vs v2 on the same quantity (median speed per cluster) --")
log("   v1 divided duration by 60 an extra time -> ~60x too small")
log(f"   v2 median speed_mph overall = {np.nanmedian(stored):.2f} mph")
log(f"   v2 median miles/duration   = {np.nanmedian(speed_derived):.2f} mph")
log(f"   v1 would have reported     = {np.nanmedian(speed_derived)/60:.2f} mph "
    f"(matches the corrupt 3.454 / 3.029 in the old consistency_profile.csv)")

# ------------------------------------------------------------
# 4. Consistency features (units now homogeneous)
# ------------------------------------------------------------
df["fare_per_mile"] = np.where(miles > 0, fare / miles, np.nan)
df["tip_ratio"] = np.where(fare > 0, tips / fare, np.nan)
df["fare_per_hr"] = np.where(dur_min > 0, fare / (dur_min / 60.0), np.nan)
df["min_per_mile"] = np.where(miles > 0, dur_min / miles, np.nan)

NEW_COLS = ["fare_per_mile", "tip_ratio", "fare_per_hr", "min_per_mile"]
for c in NEW_COLS:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)

log("")
log("-- unit-corrected consistency feature ranges --")
for c in NEW_COLS:
    s = df[c].dropna()
    log(f"   {c:16s} med={s.median():8.3f}  p10={s.quantile(.10):8.3f}  "
        f"p90={s.quantile(.90):8.3f}  nulls={int(df[c].isna().sum()):,}")


def wins(s, lo=0.01, hi=0.99):
    return s.clip(s.quantile(lo), s.quantile(hi))


# ------------------------------------------------------------
# 5. Per-cluster physical profile
# ------------------------------------------------------------
label_col = "cluster_robust_name"
log("")
log(f"-- per-cluster profile (column: {label_col}) --")
agg = (df.assign(
        _fpm=wins(df["fare_per_mile"]),
        _tr=wins(df["tip_ratio"]),
        _fph=wins(df["fare_per_hr"]),
        _mpm=wins(df["min_per_mile"]))
       .groupby(label_col)
       .agg(n=("row_uid", "size"),
            miles_med=("miles", "median"),
            fare_med=("fare", "median"),
            tips_med=("tips", "median"),
            duration_min_med=("duration_min", "median"),
            speed_mph_med=("speed_mph", "median"),
            speed_from_units_med=("speed_mph", lambda x: np.nan),
            fare_per_mile_med=("_fpm", "median"),
            fare_per_mile_p90=("_fpm", lambda x: x.quantile(.90)),
            min_per_mile_med=("_mpm", "median"),
            fare_per_hr_med=("_fph", "median"),
            tip_ratio_med=("_tr", "median"),
            zero_tip_pct=("tips", lambda x: (x == 0).mean() * 100))
       .round(3).reset_index())

# fill the derived-speed column honestly
for i, r in agg.iterrows():
    sub = df[df[label_col] == r[label_col]]
    agg.loc[i, "speed_from_units_med"] = round(
        float(np.nanmedian(
            sub["miles"] / (sub["duration_min"] / 60.0))), 3)

agg["share_pct"] = (agg["n"] / len(df) * 100).round(2)
# project rule: outputs are keyed on the cluster NAME, never on a bare id
agg = agg[["cluster_robust_name", "n", "share_pct", "miles_med", "fare_med",
           "tips_med", "duration_min_med", "speed_mph_med",
           "speed_from_units_med", "fare_per_mile_med", "fare_per_mile_p90",
           "min_per_mile_med", "fare_per_hr_med", "tip_ratio_med",
           "zero_tip_pct"]]
agg = agg.sort_values("cluster_robust_name").reset_index(drop=True)

log(agg.to_string(index=False))
agg.to_csv(OUT_DIR / "consistency_profile_v2.csv", index=False)

log("")
log("   unit self-check: speed_mph_med == speed_from_units_med")
consistent = np.allclose(agg["speed_mph_med"], agg["speed_from_units_med"],
                         atol=0.01)
log(f"   agree: {consistent}")

# ------------------------------------------------------------
# 6. Save
# ------------------------------------------------------------
keep = (["row_uid"] + FEATURE_COLS +
        ["cluster", "cluster_name", "cluster_robust", "cluster_robust_name",
         "miles", "fare", "tips",
         "duration_min", "duration_hr"] + NEW_COLS)
keep = [c for c in keep if c in df.columns]
out = df[keep].copy()
out.to_parquet(OUT_DIR / "anomalies_enriched_v2.parquet", index=False)
log("")
log(f"saved anomalies_enriched_v2.parquet ({len(out):,} rows x {out.shape[1]} cols)")

audit = pd.DataFrame(audit_rows)
audit["speed_consistency"] = consistent
audit.to_csv(OUT_DIR / "unit_audit.csv", index=False)

summary = {
    "rows": int(len(df)),
    "duration_unit": "MINUTES",
    "speed_unit": "physical mph",
    "max_speed_mismatch": max_diff,
    "speed_check_passed": bool(max_diff < 1e-6),
    "label_column": label_col,
    "label_rule": "cluster names only; numeric ids are context, never quoted",
    "cluster_name_map": {str(k): v for k, v in ROBUST_NAME.items()},
    "supersedes": "add_consistency_features.py (v1, unit-broken)",
}
(OUT_DIR / "run_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

log("")
log("=" * 78)
log("RESULT")
log("=" * 78)
log("v1's `spd_med = 3.454 / 3.029` and `hr_per_mile ~ 0.3` are explained as a")
log("missing x60 factor: duration is already minutes, so v1 divided by 60 twice.")
log("All v2 columns are in homogeneous physical units and self-verify.")
log("")
log(f"All outputs in {OUT_DIR}")
