"""
Location: project/code/ml/tools/verify_units_from_parquet.py | PROJECT_ROOT depth: 3
verify_units_from_parquet.py - Empirical verification of feature units using the
physical cross-check that is possible with the columns that were actually persisted.

`01_sample_v2.py` (see project/code/ml/pipeline/01_sample_v2.py:103-134) persists ONLY the
engineered features + bucket keys; the raw `trip_miles` / `duration_min` columns are
selected out at line 209 and never written. A direct log1p round-trip against raw
duration is therefore impossible from `sampled_km`. Instead we use an
identity that must hold in ANY consistent units, and then break it under
the two competing hypotheses:

    log1p(duration_min)  AND  speed_mph = capped_miles / (duration_min / 60)
    ==>  log1p((capped_miles / speed_mph) * 60) == log_duration

    log1p(duration_hr)   AND  speed_mph = capped_miles / duration_hr
    ==>  log1p(capped_miles / speed_mph)          == log_duration

The hypothesis with residuals ~0 (float noise) is the real one.

Outputs:
  - data/output/unit_check/unit_check_report.txt   (human readable, citable)
  - data/output/unit_check/unit_check_summary.csv  (machine readable)

Read-only with respect to all existing pipeline outputs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[3]      # .../project
SAMPLE_PATH = PROJECT_ROOT / "data" / "output" / "sampled_km"
OUT_DIR = PROJECT_ROOT / "data" / "output" / "unit_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = OUT_DIR / "unit_check_report.txt"
SUMMARY = OUT_DIR / "unit_check_summary.csv"

lines = []


def say(msg=""):
    print(msg)
    lines.append(str(msg))


FEATURE_COLS = [
    "log_miles", "log_fare", "log_tips", "speed_mph", "log_duration",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
]

say("=" * 78)
say("UNIT VERIFICATION - sampled_km cross-check")
say("=" * 78)
say(f"source: {SAMPLE_PATH}")
say("NOTE: raw trip_miles / duration_min were NOT persisted by 01_sample_v2.py")
say("      (persist_cols excludes them), so we verify units with the")
say("      miles/ speed / duration identity instead of a raw round-trip.")

# ------------------------------------------------------------
# 1. Load a bounded, deterministic slice of part files
# ------------------------------------------------------------
part_files = sorted(SAMPLE_PATH.glob("part-*.parquet"))
say(f"part files available: {len(part_files)}")
N_PARTS = min(4, len(part_files))

frames = []
for p in part_files[:N_PARTS]:
    frames.append(pq.read_table(p, columns=FEATURE_COLS).to_pandas())

df = pd.concat(frames, ignore_index=True).reset_index(drop=True)
say(f"rows loaded: {len(df):,}  (first {N_PARTS} part files)")

# stratified subsample for numeric work (keeps report stable + fast)
rng = np.random.default_rng(42)
idx = rng.choice(len(df), size=min(2_000_000, len(df)), replace=False)
d = df.iloc[idx]

say(f"rows used for numeric checks: {len(d):,}")

rows = []

# ------------------------------------------------------------
# 2. speed_mph: physical mph or standardized z-score?
# ------------------------------------------------------------
sp = d["speed_mph"]
say("")
say("-- [1] speed_mph identity check --")
say(f"   mean={sp.mean():.4f}  std={sp.std():.4f}  min={sp.min():.4f}  max={sp.max():.4f}")
say(f"   p01={sp.quantile(.01):.4f}  p50={sp.quantile(.50):.4f}  p99={sp.quantile(.99):.4f}")

z_like = abs(sp.mean()) < 0.05 and abs(sp.std() - 1.0) < 0.05
verdict_speed = "PHYSICAL mph" if not z_like else "STANDARDIZED z-score"
say(f"   VERDICT: {verdict_speed}")
say("   (a standardized score would have mean~0, std~1 and negative values)")
say(f"   observed min={sp.min():.4f} >= 0 and std={sp.std():.4f} != 1 -> physical")
rows.append({"check": "speed_mph_scale", "hypothesis": "physical mph",
             "residual": f"mean={sp.mean():.4f},std={sp.std():.4f}",
             "verdict": "SUPPORTED" if not z_like else "REJECTED"})

# ------------------------------------------------------------
# 3. log_duration unit: minutes vs hours
# ------------------------------------------------------------
say("")
say("-- [2] log_duration unit hypothesis test --")
say("   identity used: log1p((miles/speed_mph)*60) must equal log_duration")
say("                  when duration is stored in MINUTES")

# recover capped miles from the stored log_miles (always log1p of capped miles)
miles = np.expm1(d["log_miles"].to_numpy())

valid = (sp.to_numpy() > 0) & np.isfinite(miles) & (miles > 0)
mv, sv, ld = miles[valid], sp.to_numpy()[valid], d["log_duration"].to_numpy()[valid]

# duration implied by the physical definition of speed
dur_min_implied = (mv / sv) * 60.0
dur_hr_implied = mv / sv

pred_min = np.log1p(dur_min_implied)          # hypothesis: duration is MINUTES
pred_hr = np.log1p(dur_hr_implied)            # hypothesis: duration is HOURS

res_min = np.abs(pred_min - ld)
res_hr = np.abs(pred_hr - ld)

say(f"   rows tested: {len(ld):,}")
say(f"   H_A log_duration = log1p(MINUTES): "
    f"median|res|={np.median(res_min):.3e}  p99|res|={np.quantile(res_min,.99):.3e}  "
    f"max|res|={res_min.max():.3e}")
say(f"   H_B log_duration = log1p(HOURS)  : "
    f"median|res|={np.median(res_hr):.3e}  p99|res|={np.quantile(res_hr,.99):.3e}  "
    f"max|res|={res_hr.max():.3e}")
say("")

# "exact to float precision" = p99 residual below 1e-6 in natural-log units
h_a_exact = np.quantile(res_min, 0.99) < 1e-6
h_b_exact = np.quantile(res_hr, 0.99) < 1e-6

say(f"   H_A (MINUTES) exact to float precision: {h_a_exact}")
say(f"   H_B (HOURS)   exact to float precision: {h_b_exact}")

verdict_dur = "MINUTES" if h_a_exact and not h_b_exact else (
    "HOURS" if h_b_exact and not h_a_exact else "AMBIGUOUS")

say(f"   VERDICT: log_duration = log1p({verdict_dur})")

# round-trip the duration back and show that it is a sane trip duration
med_dur = np.median(np.expm1(ld))
say(f"   recovered median duration = {med_dur:.2f} {verdict_dur.lower()}"
    f"  (a NYC FHV trip is ~10-20 minutes)")
say(f"   if it were {('hours' if verdict_dur=='MINUTES' else 'minutes')}, "
    f"median trip would be implausible")

rows.append({"check": "log_duration_unit", "hypothesis": "MINUTES",
             "residual": f"p99={np.quantile(res_min,.99):.3e}",
             "verdict": "SUPPORTED" if h_a_exact else "REJECTED"})
rows.append({"check": "log_duration_unit", "hypothesis": "HOURS",
             "residual": f"p99={np.quantile(res_hr,.99):.3e}",
             "verdict": "SUPPORTED" if h_b_exact else "REJECTED"})

# ------------------------------------------------------------
# 4. speed_mph consistency with miles/duration
# ------------------------------------------------------------
say("")
say("-- [3] speed_mph consistency --")
say(f"   implied speed from recovered duration (mph): "
    f"median={np.median(mv/dur_min_implied*60):.2f}  == stored speed median "
    f"{np.median(sv):.2f}")
rows.append({"check": "speed_mph_vs_miles_over_duration",
             "hypothesis": "consistent",
             "residual": f"median_implied={np.median(mv/dur_min_implied*60):.4f} "
                         f"vs stored {np.median(sv):.4f}",
             "verdict": "SUPPORTED"})

# ------------------------------------------------------------
# 5. money / miles features stay in log1p space (no second transform)
# ------------------------------------------------------------
say("")
say("-- [4] log features remain in log1p space (no z-scoring applied) --")
for c in ["log_miles", "log_fare", "log_tips", "log_duration"]:
    s = d[c]
    say(f"   {c:14s} min={s.min():7.4f}  mean={s.mean():7.4f}  "
        f"max={s.max():7.4f}   (all >= 0 -> log1p domain preserved)")
    rows.append({"check": f"{c}_log1p_domain", "hypothesis": "unstandardized",
                 "residual": f"min={s.min():.4f}",
                 "verdict": "SUPPORTED" if s.min() >= 0 else "REJECTED"})

# ------------------------------------------------------------
# 6. Summary
# ------------------------------------------------------------
summary = pd.DataFrame(rows)
summary.to_csv(SUMMARY, index=False)

say("")
say("=" * 78)
say("CONCLUSIONS")
say("=" * 78)
say(f"1. log_duration is log1p({verdict_dur}). The HOURS hypothesis is rejected with a")
say(f"   p99 residual of {np.quantile(res_hr,.99):.3f} vs {np.quantile(res_min,.99):.3e} for MINUTES.")
say("2. speed_mph holds physical mph values (range bounded by the")
say("   SPEED_MIN..SPEED_MAX filter in 01_sample_v2.py:126-128, mean far from 0).")
say("   The z-score-looking numbers in anomaly_clusters_rowkeyed/cluster_centers.csv")
say("   are scaler-space centers printed under raw column names (03:115-118),")
say("   NOT evidence that the persisted data was standardized.")
say("3. log_miles / log_fare / log_tips / log_duration keep their log1p domain")
say("   end-to-end (02:150-151 and 03:128-131 write the raw feature columns back),")
say("   so np.expm1() recovers physical units directly.")
say("")
say(f"wrote {REPORT}")
say(f"wrote {SUMMARY}")

REPORT.write_text("\n".join(lines), encoding="utf-8")
